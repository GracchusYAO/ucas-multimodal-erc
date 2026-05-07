"""把评估结果整理成课程报告可以直接使用的图。"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")  # 防止 matplotlib 写用户目录失败

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


MODEL_ORDER = ["text_only", "concat_tav", "dgf", "dgf_dropout", "dgf_context"]
MODEL_NAMES = {
    "text_only": "Text",
    "concat_tav": "Concat",
    "dgf": "DGF",
    "dgf_dropout": "DGF+Drop",
    "dgf_context": "DGF+Ctx",
}
EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
GATE_COLUMNS = ["gate_text", "gate_audio", "gate_visual"]
GATE_NAMES = ["Text", "Audio", "Visual"]


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_metrics(evaluate_root: Path, models: list[str]) -> dict[str, dict]:
    """读取 results/evaluate/{model}_test/metrics.json。"""
    metrics = {}
    for model in models:
        path = evaluate_root / f"{model}_test" / "metrics.json"
        if path.exists():
            metrics[model] = load_json(path)
    if not metrics:
        raise FileNotFoundError(f"No metrics.json found under {evaluate_root}")
    return metrics


def save_metrics_summary(output_dir: Path, metrics: dict[str, dict]) -> None:
    path = output_dir / "metrics_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["model", "accuracy", "weighted_f1", "macro_f1"])
        for model, item in metrics.items():
            writer.writerow([model, item["accuracy"], item["weighted_f1"], item["macro_f1"]])


def plot_f1_comparison(output_dir: Path, metrics: dict[str, dict]) -> None:
    models = list(metrics)
    labels = [MODEL_NAMES.get(model, model) for model in models]
    weighted = [metrics[model]["weighted_f1"] for model in models]
    macro = [metrics[model]["macro_f1"] for model in models]

    x_positions = list(range(len(models)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([x - width / 2 for x in x_positions], weighted, width, label="Weighted F1", color="#4C78A8")
    ax.bar([x + width / 2 for x in x_positions], macro, width, label="Macro F1", color="#F58518")

    ax.set_ylim(0, max(weighted + macro) + 0.08)
    ax.set_xticks(x_positions, labels=labels)
    ax.set_ylabel("Score")
    ax.set_title("MELD Test F1 Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    for x, value in zip([x - width / 2 for x in x_positions], weighted):
        ax.text(x, value + 0.008, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    for x, value in zip([x + width / 2 for x in x_positions], macro):
        ax.text(x, value + 0.008, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "f1_comparison.png", dpi=200)
    plt.close()


def plot_per_class_f1(output_dir: Path, model_name: str, metrics: dict) -> None:
    values = [metrics["per_class"][emotion]["f1"] for emotion in EMOTIONS]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(EMOTIONS, values, color="#54A24B")
    ax.set_ylim(0, max(values) + 0.12)
    ax.set_ylabel("F1")
    ax.set_title(f"Per-class F1 ({MODEL_NAMES.get(model_name, model_name)})")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=30)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.008, f"{value:.3f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / f"per_class_f1_{model_name}.png", dpi=200)
    plt.close()


def load_confusion_matrix(path: Path) -> tuple[list[str], list[list[float]]]:
    with path.open(encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader)
        labels = header[1:]
        matrix = [[float(value) for value in row[1:]] for row in reader]
    return labels, matrix


def normalize_rows(matrix: list[list[float]]) -> list[list[float]]:
    """按真实类别归一化，能看出每类主要被错分到哪里。"""
    normalized = []
    for row in matrix:
        total = sum(row)
        normalized.append([value / total if total else 0.0 for value in row])
    return normalized


def plot_confusion_matrix(output_dir: Path, evaluate_root: Path, model_name: str) -> None:
    source = evaluate_root / f"{model_name}_test" / "confusion_matrix.csv"
    labels, matrix = load_confusion_matrix(source)
    normalized = normalize_rows(matrix)

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(image, ax=ax)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.set_title(f"Normalized Confusion Matrix ({MODEL_NAMES.get(model_name, model_name)})")

    for row_index, row in enumerate(normalized):
        for col_index, value in enumerate(row):
            ax.text(col_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / f"confusion_matrix_{model_name}.png", dpi=200)
    plt.close()


def load_gate_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def average_gates_by_emotion(rows: list[dict[str, str]]) -> dict[str, list[float]]:
    totals = {emotion: [0.0, 0.0, 0.0] for emotion in EMOTIONS}
    counts = {emotion: 0 for emotion in EMOTIONS}
    for row in rows:
        emotion = row["gold"]
        if emotion not in totals:
            continue
        for index, column in enumerate(GATE_COLUMNS):
            totals[emotion][index] += float(row[column])
        counts[emotion] += 1

    averages = {}
    for emotion in EMOTIONS:
        count = max(counts[emotion], 1)
        averages[emotion] = [value / count for value in totals[emotion]]
    return averages


def save_gate_summary(output_dir: Path, model_name: str, averages: dict[str, list[float]]) -> None:
    path = output_dir / f"gate_weights_by_emotion_{model_name}.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["emotion", *GATE_COLUMNS])
        for emotion in EMOTIONS:
            writer.writerow([emotion, *averages[emotion]])


def plot_gate_weights(output_dir: Path, evaluate_root: Path, model_name: str) -> None:
    source = evaluate_root / f"{model_name}_test" / "gate_weights.csv"
    if not source.exists():
        print(f"skip gate plot: missing {source}")
        return

    rows = load_gate_rows(source)
    averages = average_gates_by_emotion(rows)
    save_gate_summary(output_dir, model_name, averages)

    text_values = [averages[emotion][0] for emotion in EMOTIONS]
    audio_values = [averages[emotion][1] for emotion in EMOTIONS]
    visual_values = [averages[emotion][2] for emotion in EMOTIONS]
    x_positions = list(range(len(EMOTIONS)))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x_positions, text_values, label=GATE_NAMES[0], color="#4C78A8")
    ax.bar(x_positions, audio_values, bottom=text_values, label=GATE_NAMES[1], color="#F58518")
    bottoms = [text + audio for text, audio in zip(text_values, audio_values)]
    ax.bar(x_positions, visual_values, bottom=bottoms, label=GATE_NAMES[2], color="#54A24B")

    ax.set_ylim(0, 1)
    ax.set_xticks(x_positions, labels=EMOTIONS, rotation=30)
    ax.set_ylabel("Average gate weight")
    ax.set_title(f"Average Modality Gate by Emotion ({MODEL_NAMES.get(model_name, model_name)})")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(output_dir / f"gate_weights_by_emotion_{model_name}.png", dpi=200)
    plt.close()


def load_missing_modality_metrics(missing_root: Path, model_name: str, full_metrics: dict) -> list[dict]:
    """读取遮掉不同模态后的评估结果。"""
    conditions = [
        ("full", None),
        ("no_text", missing_root / f"{model_name}_no_text" / "metrics.json"),
        ("no_audio", missing_root / f"{model_name}_no_audio" / "metrics.json"),
        ("no_visual", missing_root / f"{model_name}_no_visual" / "metrics.json"),
        ("no_audio_visual", missing_root / f"{model_name}_no_audio_visual" / "metrics.json"),
    ]

    rows = []
    for condition, path in conditions:
        if path is None:
            metrics = full_metrics
        elif path.exists():
            metrics = load_json(path)
        else:
            continue
        rows.append(
            {
                "condition": condition,
                "accuracy": metrics["accuracy"],
                "weighted_f1": metrics["weighted_f1"],
                "macro_f1": metrics["macro_f1"],
            }
        )
    return rows


def plot_missing_modality_analysis(
    output_dir: Path,
    missing_root: Path,
    model_name: str,
    full_metrics: dict,
) -> None:
    rows = load_missing_modality_metrics(missing_root, model_name, full_metrics)
    if len(rows) <= 1:
        print(f"skip missing-modality plot: no ablation metrics under {missing_root}")
        return

    csv_path = output_dir / f"missing_modality_metrics_{model_name}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["condition", "accuracy", "weighted_f1", "macro_f1"])
        writer.writeheader()
        writer.writerows(rows)

    labels = [row["condition"].replace("_", "\n") for row in rows]
    weighted = [row["weighted_f1"] for row in rows]
    macro = [row["macro_f1"] for row in rows]
    x_positions = list(range(len(rows)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([x - width / 2 for x in x_positions], weighted, width, label="Weighted F1", color="#4C78A8")
    ax.bar([x + width / 2 for x in x_positions], macro, width, label="Macro F1", color="#E45756")
    ax.set_ylim(0, max(weighted + macro) + 0.08)
    ax.set_xticks(x_positions, labels=labels)
    ax.set_ylabel("Score")
    ax.set_title(f"Missing Modality Analysis ({MODEL_NAMES.get(model_name, model_name)})")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(output_dir / f"missing_modality_analysis_{model_name}.png", dpi=200)
    plt.close()


def choose_best_model(metrics: dict[str, dict]) -> str:
    """默认用 test weighted F1 最高的模型做细节图。"""
    return max(metrics, key=lambda model: metrics[model]["weighted_f1"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize MELD experiment results.")
    parser.add_argument("--evaluate-root", default="results/evaluate")
    parser.add_argument("--missing-root", default="results/evaluate_missing")
    parser.add_argument("--output-dir", default="results/visualizations")
    parser.add_argument("--models", nargs="*", default=MODEL_ORDER)
    parser.add_argument("--best-model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_root = Path(args.evaluate_root)
    missing_root = Path(args.missing_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(evaluate_root, args.models)
    best_model = args.best_model or choose_best_model(metrics)

    save_metrics_summary(output_dir, metrics)
    plot_f1_comparison(output_dir, metrics)
    plot_per_class_f1(output_dir, best_model, metrics[best_model])
    plot_confusion_matrix(output_dir, evaluate_root, best_model)
    plot_gate_weights(output_dir, evaluate_root, best_model)
    plot_missing_modality_analysis(output_dir, missing_root, best_model, metrics[best_model])

    print(f"best_model={best_model}")
    print(f"saved visualizations: {output_dir}")


if __name__ == "__main__":
    main()
