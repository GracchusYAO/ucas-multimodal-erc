"""Evaluate an ensemble from cached dev/test logits.

This avoids reloading large text or feature models when we only need to test
whether a newly trained branch provides complementary predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import torch

from src.dataset import ID2EMOTION


def load_payload(logit_dir: str | Path, split: str) -> dict:
    path = Path(logit_dir) / f"{split}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing logits: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def load_all(logit_dirs: list[str], split: str) -> tuple[list[torch.Tensor], torch.Tensor, list[str], list[str]]:
    logits_list = []
    labels = None
    keys = None
    names = []

    for logit_dir in logit_dirs:
        payload = load_payload(logit_dir, split)
        current_labels = payload["labels"].long()
        current_keys = list(payload["keys"])
        if labels is None:
            labels = current_labels
            keys = current_keys
        elif not torch.equal(labels, current_labels) or keys != current_keys:
            raise ValueError(f"Prediction order mismatch: {logit_dir}")

        logits_list.append(payload["logits"].float())
        names.append(payload.get("name") or Path(logit_dir).name)
        print(f"loaded {split}: {logit_dir} {tuple(payload['logits'].shape)}")

    if labels is None or keys is None:
        raise ValueError("At least one --logit-dir is required.")
    return logits_list, labels, keys, names


def average_logits(logits_list: list[torch.Tensor], average: str, weights: torch.Tensor | None) -> torch.Tensor:
    stacked = torch.stack(logits_list)  # [model_count, N, C]
    if average == "prob":
        stacked = torch.softmax(stacked, dim=-1)
    if weights is None:
        return stacked.mean(dim=0)
    return (stacked * weights.view(-1, 1, 1)).sum(dim=0)


def build_metrics(labels: torch.Tensor, predictions: torch.Tensor) -> dict[str, float]:
    labels = labels.long()
    predictions = predictions.long()
    accuracy = float((labels == predictions).float().mean())
    total = int(labels.numel())

    macro_f1 = 0.0
    weighted_f1 = 0.0
    for class_id in range(len(ID2EMOTION)):
        gold = labels == class_id
        pred = predictions == class_id
        tp = int((gold & pred).sum())
        fp = int((~gold & pred).sum())
        fn = int((gold & ~pred).sum())
        support = int(gold.sum())

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        macro_f1 += f1
        weighted_f1 += f1 * support / total

    return {
        "accuracy": accuracy,
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1 / len(ID2EMOTION),
    }


def search_weights(
    logits_list: list[torch.Tensor],
    labels: torch.Tensor,
    average: str,
    trials: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    rng = random.Random(seed)
    model_count = len(logits_list)
    candidates = [torch.ones(model_count) / model_count]
    candidates.extend(torch.eye(model_count))

    for _ in range(trials):
        raw = torch.tensor([rng.random() for _ in range(model_count)], dtype=torch.float32)
        candidates.append(raw / raw.sum().clamp_min(1e-8))

    best_weights = candidates[0]
    best_metrics = build_metrics(labels, average_logits(logits_list, average, best_weights).argmax(dim=1))
    for weights in candidates[1:]:
        predictions = average_logits(logits_list, average, weights).argmax(dim=1)
        metrics = build_metrics(labels, predictions)
        if metrics["weighted_f1"] > best_metrics["weighted_f1"]:
            best_weights = weights
            best_metrics = metrics
    return best_weights, best_metrics


def save_predictions(path: Path, keys: list[str], labels: torch.Tensor, predictions: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["key", "gold_id", "gold", "pred_id", "pred", "correct"])
        for key, gold, pred in zip(keys, labels.tolist(), predictions.tolist()):
            writer.writerow([key, gold, ID2EMOTION[gold], pred, ID2EMOTION[pred], int(gold == pred)])


def format_weights(names: list[str], weights: torch.Tensor) -> list[str]:
    return [f"{name}:{weight:.4f}" for name, weight in zip(names, weights.tolist())]


def evaluate(args: argparse.Namespace) -> dict:
    weights = None
    tune_metrics = None
    if args.tune_weights_on_dev:
        dev_logits, dev_labels, _, names = load_all(args.logit_dir, "dev")
        weights, tune_metrics = search_weights(
            dev_logits,
            dev_labels,
            args.average,
            args.weight_search_trials,
            args.seed,
        )
        print(
            "tuned dev: "
            f"weighted_f1={tune_metrics['weighted_f1']:.4f} "
            f"macro_f1={tune_metrics['macro_f1']:.4f} "
            f"weights={format_weights(names, weights)}"
        )

    logits_list, labels, keys, names = load_all(args.logit_dir, args.split)
    scores = average_logits(logits_list, args.average, weights)
    predictions = scores.argmax(dim=1)
    metrics = build_metrics(labels, predictions)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_predictions(output_dir / "predictions.csv", keys, labels, predictions)

    result = {
        "split": args.split,
        "average": args.average,
        "model_names": names,
        "ensemble_weights": weights.tolist() if weights is not None else None,
        "tune_dev_metrics": tune_metrics,
        **metrics,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print(
        f"logits ensemble {args.split}: "
        f"accuracy={metrics['accuracy']:.4f} "
        f"weighted_f1={metrics['weighted_f1']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )
    print(f"saved {output_dir}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate cached logits ensemble.")
    parser.add_argument("--logit-dir", action="append", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default="results/logits_ensemble")
    parser.add_argument("--average", choices=("logit", "prob"), default="prob")
    parser.add_argument("--tune-weights-on-dev", action="store_true")
    parser.add_argument("--weight-search-trials", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=114514)
    return parser.parse_args()


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
