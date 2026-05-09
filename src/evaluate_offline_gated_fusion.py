"""离线置信门控融合：强文本主干 + 多模态辅助。

核心想法：
1. fine-tuned text 已经明显强于冻结特征多模态，所以它应该是主干。
2. audio/visual 当前比较噪声，不能无条件平均进去。
3. 只在文本置信度较低的样本上，让多模态 logits 以一个较小权重介入。

这样仍然是 gated multimodal，但门控是保守的：多模态负责补充，不负责抢方向盘。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from src.dataset import ID2EMOTION, MELD_SPLITS
from src.evaluate_text_ensemble import average_logits, build_metrics, search_ensemble_weights
from src.train_text_finetune import set_seed


def load_payload(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"logits", "labels", "keys"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"{path} missing fields: {sorted(missing)}")
    return payload


def load_split(logit_dirs: list[str], split: str) -> tuple[list[str], list[torch.Tensor], torch.Tensor, list[str]]:
    """读取若干 logits 目录，并确认样本顺序一致。"""
    names: list[str] = []
    logits_list: list[torch.Tensor] = []
    reference_labels: torch.Tensor | None = None
    reference_keys: list[str] | None = None

    for item in logit_dirs:
        path = Path(item) / f"{split}.pt"
        payload = load_payload(path)
        logits = payload["logits"].float()
        labels = payload["labels"].long()
        keys = list(payload["keys"])
        name = str(payload.get("name") or Path(item).name)

        if reference_labels is None or reference_keys is None:
            reference_labels = labels
            reference_keys = keys
        elif not torch.equal(reference_labels, labels) or reference_keys != keys:
            raise ValueError(f"Prediction order mismatch: {path}")

        names.append(name)
        logits_list.append(logits)

    if reference_labels is None or reference_keys is None:
        raise ValueError("At least one logits directory is required.")
    return names, logits_list, reference_labels, reference_keys


def to_probs(logits_list: list[torch.Tensor], average: str, weights: torch.Tensor) -> torch.Tensor:
    """把若干 logits 按权重合成概率分布。"""
    scores = average_logits(logits_list, average, weights)
    if average == "prob":
        return scores.clamp_min(0.0)
    return torch.softmax(scores, dim=1)


def make_weight_candidates(count: int, trials: int, seed: int) -> list[torch.Tensor]:
    """给辅助多模态模型生成一些候选权重：平均、单模型、随机。"""
    generator = torch.Generator().manual_seed(seed)
    candidates = [torch.full((count,), 1.0 / count)]
    candidates.extend(torch.eye(count))
    if trials > 0:
        random_weights = torch.rand(trials, count, generator=generator)
        random_weights = random_weights / random_weights.sum(dim=1, keepdim=True)
        candidates.extend(random_weights)
    return candidates


def fast_weighted_f1(labels: torch.Tensor, predictions: torch.Tensor) -> float:
    """搜索阶段只需要 weighted F1，用混淆矩阵比反复手写循环快很多。"""
    num_classes = len(ID2EMOTION)
    flat_index = labels.long() * num_classes + predictions.long()
    matrix = torch.bincount(flat_index, minlength=num_classes * num_classes).float()
    matrix = matrix.view(num_classes, num_classes)

    tp = matrix.diag()
    support = matrix.sum(dim=1)
    predicted = matrix.sum(dim=0)
    precision = tp / predicted.clamp(min=1.0)
    recall = tp / support.clamp(min=1.0)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
    return float((f1 * support).sum().item() / support.sum().clamp(min=1.0).item())


def gated_probs(
    text_probs: torch.Tensor,
    aux_probs: torch.Tensor,
    threshold: float,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """文本低置信时打开多模态门；alpha 是多模态最多能占的权重。"""
    confidence = text_probs.max(dim=1).values
    gate_aux = (confidence < threshold).float() * alpha  # 低置信样本才引入多模态
    fused = text_probs * (1.0 - gate_aux).unsqueeze(1) + aux_probs * gate_aux.unsqueeze(1)
    return fused, gate_aux


def gated_probs_classwise(
    text_probs: torch.Tensor,
    aux_probs: torch.Tensor,
    thresholds: torch.Tensor,
    alphas: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """按文本预测类别设置 gate；弱类别可以更积极地看多模态。"""
    text_pred = text_probs.argmax(dim=1)
    confidence = text_probs.max(dim=1).values
    threshold_per_sample = thresholds[text_pred]  # 每条样本用自己预测类别的阈值
    alpha_per_sample = alphas[text_pred]  # 每条样本用自己预测类别的辅助权重上限
    gate_aux = (confidence < threshold_per_sample).float() * alpha_per_sample
    fused = text_probs * (1.0 - gate_aux).unsqueeze(1) + aux_probs * gate_aux.unsqueeze(1)
    return fused, gate_aux


def search_gate(
    dev_text_probs: torch.Tensor,
    dev_aux_logits: list[torch.Tensor],
    labels: torch.Tensor,
    aux_average: str,
    args: argparse.Namespace,
) -> tuple[dict, torch.Tensor]:
    """在 dev 上搜索多模态权重、置信阈值和门控强度。"""
    thresholds = [round(args.threshold_min + i * args.threshold_step, 4) for i in range(args.threshold_count)]
    alphas = [round(args.alpha_min + i * args.alpha_step, 4) for i in range(args.alpha_count)]
    if args.include_zero_alpha and 0.0 not in alphas:
        alphas = [0.0] + alphas

    best: dict | None = None
    best_score = -1.0
    best_aux_weights: torch.Tensor | None = None
    aux_candidates = make_weight_candidates(len(dev_aux_logits), args.aux_weight_search_trials, args.seed + 17)

    for aux_weights in aux_candidates:
        aux_probs = to_probs(dev_aux_logits, aux_average, aux_weights)
        for threshold in thresholds:
            for alpha in alphas:
                scores, gate_aux = gated_probs(dev_text_probs, aux_probs, threshold, alpha)
                predictions = scores.argmax(dim=1)
                score = fast_weighted_f1(labels, predictions)
                if score > best_score:
                    metrics = build_metrics(labels, predictions)
                    best_score = score
                    best = {
                        "threshold": threshold,
                        "alpha": alpha,
                        "metrics": metrics,
                        "mean_gate_aux": float(gate_aux.mean().item()),
                        "active_ratio": float((gate_aux > 0).float().mean().item()),
                    }
                    best_aux_weights = aux_weights.clone()

    assert best is not None and best_aux_weights is not None
    return best, best_aux_weights


def search_class_gate(
    dev_text_probs: torch.Tensor,
    dev_aux_probs: torch.Tensor,
    labels: torch.Tensor,
    base_gate: dict,
    args: argparse.Namespace,
) -> dict:
    """在全局 gate 的基础上，按文本预测类别做一小轮坐标搜索。"""
    num_classes = len(ID2EMOTION)
    thresholds_grid = [round(args.threshold_min + i * args.threshold_step, 4) for i in range(args.threshold_count)]
    alphas_grid = [round(args.alpha_min + i * args.alpha_step, 4) for i in range(args.alpha_count)]
    if 0.0 not in alphas_grid:
        alphas_grid = [0.0] + alphas_grid  # 某些类别如果多模态没帮助，就允许关掉

    thresholds = torch.full((num_classes,), float(base_gate["threshold"]))
    alphas = torch.full((num_classes,), float(base_gate["alpha"]))

    scores, gate_aux = gated_probs_classwise(dev_text_probs, dev_aux_probs, thresholds, alphas)
    predictions = scores.argmax(dim=1)
    best_score = fast_weighted_f1(labels, predictions)

    for _ in range(args.class_gate_rounds):
        changed = False
        for class_id in range(num_classes):
            class_threshold = float(thresholds[class_id].item())
            class_alpha = float(alphas[class_id].item())
            class_best_score = best_score

            for threshold in thresholds_grid:
                for alpha in alphas_grid:
                    trial_thresholds = thresholds.clone()
                    trial_alphas = alphas.clone()
                    trial_thresholds[class_id] = threshold
                    trial_alphas[class_id] = alpha
                    scores, _ = gated_probs_classwise(dev_text_probs, dev_aux_probs, trial_thresholds, trial_alphas)
                    predictions = scores.argmax(dim=1)
                    score = fast_weighted_f1(labels, predictions)
                    if score > class_best_score:
                        class_best_score = score
                        class_threshold = threshold
                        class_alpha = alpha

            if class_best_score > best_score:
                thresholds[class_id] = class_threshold
                alphas[class_id] = class_alpha
                best_score = class_best_score
                changed = True

        if not changed:
            break

    scores, gate_aux = gated_probs_classwise(dev_text_probs, dev_aux_probs, thresholds, alphas)
    predictions = scores.argmax(dim=1)
    metrics = build_metrics(labels, predictions)
    return {
        "mode": "class",
        "thresholds": thresholds.tolist(),
        "alphas": alphas.tolist(),
        "metrics": metrics,
        "mean_gate_aux": float(gate_aux.mean().item()),
        "active_ratio": float((gate_aux > 0).float().mean().item()),
    }


def save_predictions(
    path: Path,
    keys: list[str],
    labels: torch.Tensor,
    predictions: torch.Tensor,
    text_probs: torch.Tensor,
    aux_probs: torch.Tensor,
    gate_aux: torch.Tensor,
) -> None:
    """保存更适合分析门控行为的预测表。"""
    text_pred = text_probs.argmax(dim=1)
    aux_pred = aux_probs.argmax(dim=1)
    text_conf = text_probs.max(dim=1).values
    aux_conf = aux_probs.max(dim=1).values

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "key",
                "gold_id",
                "gold",
                "pred_id",
                "pred",
                "correct",
                "text_pred",
                "aux_pred",
                "text_confidence",
                "aux_confidence",
                "gate_text",
                "gate_multimodal",
            ]
        )
        for index, key in enumerate(keys):
            gold = int(labels[index].item())
            pred = int(predictions[index].item())
            writer.writerow(
                [
                    key,
                    gold,
                    ID2EMOTION[gold],
                    pred,
                    ID2EMOTION[pred],
                    int(gold == pred),
                    ID2EMOTION[int(text_pred[index].item())],
                    ID2EMOTION[int(aux_pred[index].item())],
                    f"{float(text_conf[index].item()):.6f}",
                    f"{float(aux_conf[index].item()):.6f}",
                    f"{float(1.0 - gate_aux[index].item()):.6f}",
                    f"{float(gate_aux[index].item()):.6f}",
                ]
            )


def evaluate(args: argparse.Namespace) -> dict:
    if not args.text_logit_dir:
        raise ValueError("--text-logit-dir is required.")
    if not args.aux_logit_dir:
        raise ValueError("--aux-logit-dir is required.")

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    text_names, dev_text_logits, dev_labels, _ = load_split(args.text_logit_dir, args.dev_split)
    aux_names, dev_aux_logits, aux_dev_labels, _ = load_split(args.aux_logit_dir, args.dev_split)
    if not torch.equal(dev_labels, aux_dev_labels):
        raise ValueError("Dev labels mismatch between text and auxiliary logits.")

    text_weights, text_tune_metrics = search_ensemble_weights(
        dev_text_logits,
        dev_labels,
        args.text_average,
        args.text_weight_search_trials,
        args.seed,
    )
    dev_text_probs = to_probs(dev_text_logits, args.text_average, text_weights)
    best_gate, aux_weights = search_gate(dev_text_probs, dev_aux_logits, dev_labels, args.aux_average, args)
    if args.gate_mode == "class":
        dev_aux_probs = to_probs(dev_aux_logits, args.aux_average, aux_weights)
        best_gate = search_class_gate(dev_text_probs, dev_aux_probs, dev_labels, best_gate, args)

    _, test_text_logits, test_labels, test_keys = load_split(args.text_logit_dir, args.split)
    _, test_aux_logits, aux_test_labels, aux_test_keys = load_split(args.aux_logit_dir, args.split)
    if not torch.equal(test_labels, aux_test_labels) or test_keys != aux_test_keys:
        raise ValueError("Test order mismatch between text and auxiliary logits.")

    test_text_probs = to_probs(test_text_logits, args.text_average, text_weights)
    test_aux_probs = to_probs(test_aux_logits, args.aux_average, aux_weights)
    if best_gate.get("mode") == "class":
        scores, gate_aux = gated_probs_classwise(
            test_text_probs,
            test_aux_probs,
            torch.tensor(best_gate["thresholds"]),
            torch.tensor(best_gate["alphas"]),
        )
    else:
        scores, gate_aux = gated_probs(
            test_text_probs,
            test_aux_probs,
            float(best_gate["threshold"]),
            float(best_gate["alpha"]),
        )
    predictions = scores.argmax(dim=1)
    metrics = build_metrics(test_labels, predictions)

    save_predictions(
        output_dir / "predictions.csv",
        test_keys,
        test_labels,
        predictions,
        test_text_probs,
        test_aux_probs,
        gate_aux,
    )

    result = {
        "split": args.split,
        "dev_split": args.dev_split,
        "text_models": text_names,
        "aux_models": aux_names,
        "text_average": args.text_average,
        "aux_average": args.aux_average,
        "text_weights": text_weights.tolist(),
        "aux_weights": aux_weights.tolist(),
        "text_tune_dev_metrics": text_tune_metrics,
        "gate_tune_dev": best_gate,
        "test_gate": {
            "mean_gate_aux": float(gate_aux.mean().item()),
            "active_ratio": float((gate_aux > 0).float().mean().item()),
        },
        **metrics,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print(
        f"offline gated fusion {args.split}: "
        f"accuracy={metrics['accuracy']:.4f} "
        f"weighted_f1={metrics['weighted_f1']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )
    print(
        "gate: "
        f"mode={best_gate.get('mode', 'global')} "
        f"test_active={result['test_gate']['active_ratio']:.4f} "
        f"test_mean_aux={result['test_gate']['mean_gate_aux']:.4f}"
    )
    if best_gate.get("mode") != "class":
        print(f"gate params: threshold={best_gate['threshold']:.4f} alpha={best_gate['alpha']:.4f}")
    print(f"saved {output_dir}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate offline confidence-gated multimodal fusion.")
    parser.add_argument("--text-logit-dir", action="append")
    parser.add_argument("--aux-logit-dir", action="append")
    parser.add_argument("--split", default="test", choices=MELD_SPLITS)
    parser.add_argument("--dev-split", default="dev", choices=MELD_SPLITS)
    parser.add_argument("--output-dir", default="results/offline_gated_multimodal")
    parser.add_argument("--seed", type=int, default=114514)
    parser.add_argument("--text-average", choices=("logit", "prob"), default="logit")
    parser.add_argument("--aux-average", choices=("logit", "prob"), default="prob")
    parser.add_argument("--text-weight-search-trials", type=int, default=3000)
    parser.add_argument("--aux-weight-search-trials", type=int, default=200)
    parser.add_argument("--threshold-min", type=float, default=0.35)
    parser.add_argument("--threshold-step", type=float, default=0.025)
    parser.add_argument("--threshold-count", type=int, default=25)
    parser.add_argument("--alpha-min", type=float, default=0.02)
    parser.add_argument("--alpha-step", type=float, default=0.02)
    parser.add_argument("--alpha-count", type=int, default=25)
    parser.add_argument("--include-zero-alpha", action="store_true")
    parser.add_argument("--gate-mode", choices=("global", "class"), default="global")
    parser.add_argument("--class-gate-rounds", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
