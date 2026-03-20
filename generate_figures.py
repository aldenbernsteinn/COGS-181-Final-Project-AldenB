"""Generate figures for the ICML report from YOLO experiment results.

Reads results from results_yolo/<experiment>/results.json and generates
comparison plots for the report.

Usage:
    python generate_figures.py              # Generate all figures
    python generate_figures.py --results_dir results_yolo
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

RESULTS_DIR = Path("results_yolo_5cls")
FIGURES_DIR = Path("report/icml2023/figures")

# Experiment groups and display names
GROUP_LABELS = {
    "baseline": "Baseline",
    "arch": "Architecture",
    "optimizer": "Optimizer",
    "lr": "Learning Rate",
    "imgsz": "Image Size",
    "augmentation": "Augmentation",
    "init": "Initialization",
    "layers": "Layers",
    "activation": "Activation",
    "pooling": "Pooling",
    "best_combo": "Best Combo",
}


def load_all_results(results_dir):
    """Load all experiment results."""
    results = []
    for exp_dir in sorted(results_dir.iterdir()):
        result_file = exp_dir / "results.json"
        if result_file.exists():
            with open(result_file) as f:
                results.append(json.load(f))
    return results


def plot_group_comparison(results, metric, ylabel, filename, title=None):
    """Bar chart comparing experiments within each group."""
    groups = {}
    for r in results:
        g = r.get("group", "other")
        if g not in groups:
            groups[g] = []
        groups[g].append(r)

    group_order = ["baseline", "arch", "optimizer", "lr", "imgsz", "augmentation", "init", "layers", "activation", "pooling", "best_combo"]

    n_groups = len(group_order)
    ncols = 4
    nrows = (n_groups + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5 * nrows))
    axes = axes.flatten()

    # Hide unused axes
    for i in range(len(group_order), len(axes)):
        axes[i].set_visible(False)

    for idx, group_name in enumerate(group_order):
        ax = axes[idx]
        if group_name not in groups:
            ax.set_visible(False)
            continue

        group_results = groups[group_name]
        names = [r["experiment"].replace(f"{group_name}_", "") for r in group_results]
        values = [r["metrics"][metric] for r in group_results]

        colors = sns.color_palette("viridis", len(names))
        bars = ax.bar(range(len(names)), values, color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(GROUP_LABELS.get(group_name, group_name))
        ax.set_ylim(0, 1)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    plt.suptitle(title or f"{ylabel} by Experiment Group", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}")


def plot_metrics_overview(results, filename="metrics_overview.png"):
    """Grouped bar chart of mAP50, precision, recall for all experiments."""
    names = [r["experiment"] for r in results]
    metrics = ["mAP50", "precision", "recall"]
    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(14, len(names) * 0.8), 6))

    for i, metric in enumerate(metrics):
        values = [r["metrics"][metric] for r in results]
        ax.bar(x + i * width, values, width, label=metric)

    ax.set_xticks(x + width)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Detection Metrics Across All Experiments")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}")


def plot_speed_vs_accuracy(results, filename="speed_vs_accuracy.png"):
    """Scatter plot of CPU speed vs mAP50."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for r in results:
        speed = r.get("speed", {}).get("cpu_ms")
        if speed is None:
            continue
        mAP = r["metrics"]["mAP50"]
        size = r.get("model_size_mb", 5) or 5
        ax.scatter(speed, mAP, s=size * 10, alpha=0.7)
        ax.annotate(r["experiment"], (speed, mAP), fontsize=7,
                    xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("CPU Inference Time (ms)")
    ax.set_ylabel("mAP50")
    ax.set_title("Speed vs Accuracy Trade-off")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}")


def plot_per_class_ap(results, filename="per_class_ap.png"):
    """Heatmap of per-class AP50 across experiments."""
    # Filter results that have per-class data
    filtered = [r for r in results if r["metrics"].get("per_class_ap50")]
    if not filtered:
        print("  No per-class AP data available, skipping per_class_ap.png")
        return

    class_names = list(filtered[0]["metrics"]["per_class_ap50"].keys())
    exp_names = [r["experiment"] for r in filtered]

    data = np.zeros((len(filtered), len(class_names)))
    for i, r in enumerate(filtered):
        for j, cls in enumerate(class_names):
            data[i, j] = r["metrics"]["per_class_ap50"].get(cls, 0)

    fig, ax = plt.subplots(figsize=(14, max(6, len(filtered) * 0.5)))
    sns.heatmap(data, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=class_names, yticklabels=exp_names,
                ax=ax, vmin=0, vmax=1)
    ax.set_title("Per-Class AP50 Across Experiments")
    ax.set_xlabel("UI Element Class")
    ax.set_ylabel("Experiment")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}")


def plot_model_size_comparison(results, filename="model_sizes.png"):
    """Bar chart of model sizes in MB."""
    filtered = [(r["experiment"], r["model_size_mb"])
                for r in results if r.get("model_size_mb")]
    if not filtered:
        print("  No model size data available, skipping model_sizes.png")
        return

    names, sizes = zip(*filtered)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("coolwarm", len(names))
    bars = ax.bar(range(len(names)), sizes, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Model Size (MB)")
    ax.set_title("Model Size Comparison")

    for bar, val in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}")


def plot_injection_classifier(filename="injection_classifier.png"):
    """Plot injection classifier training curves if results exist."""
    results_path = Path("weights/injection_classifier/results.json")
    if not results_path.exists():
        print("  No injection classifier results, skipping injection_classifier.png")
        return

    with open(results_path) as f:
        results = json.load(f)

    history = results.get("history", {})
    if not history:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-o", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Injection Classifier: Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["val_acc"], "g-o", label="Val Accuracy")
    ax2.plot(epochs, history["val_f1"], "m-o", label="Val F1")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title("Injection Classifier: Accuracy & F1")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.suptitle(
        f"Test Acc: {results['test_acc']:.4f} | "
        f"Test F1: {results['test_f1']:.4f} | "
        f"Speed: {results['inference_speed_ms']:.1f} ms",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}")


def main():
    parser = argparse.ArgumentParser(description="Generate report figures")
    parser.add_argument("--results_dir", type=str, default=str(RESULTS_DIR))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    results = load_all_results(results_dir)
    print(f"  Found {len(results)} experiments")

    if not results:
        print("No results found. Run train_yolo.py first.")
        return

    print("\nGenerating figures...")
    sns.set_theme(style="whitegrid")

    plot_group_comparison(results, "mAP50", "mAP@0.5", "map50_comparison.png")
    plot_group_comparison(results, "recall", "Recall", "recall_comparison.png")
    plot_metrics_overview(results)
    plot_speed_vs_accuracy(results)
    plot_per_class_ap(results)
    plot_model_size_comparison(results)
    plot_injection_classifier()

    print(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
