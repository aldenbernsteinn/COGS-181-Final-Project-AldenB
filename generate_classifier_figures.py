"""Generate benchmark comparison figures for the classifier section of the report."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from PIL import Image
from pathlib import Path

FIGURES_DIR = Path("report/icml2023/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

LOGOS_DIR = Path("logos")

# Colors
COLOR_OURS = '#2563EB'      # Blue
COLOR_PAI = '#6B7280'       # Gray
COLOR_META = '#D97706'      # Amber
COLOR_PIGUARD = '#059669'   # Emerald

# Model info: (name, color, logo_file, param_count)
MODELS = [
    ('VIGIL\n(Ours, 22M)', COLOR_OURS, 'arvel-logo.png'),
    ('ProtectAI\n(200M)', COLOR_PAI, 'protectai-logo.png'),
    ('Meta Guard\n(86M)', COLOR_META, 'meta-logo.png'),
]


def load_logo_img(name, size=28):
    """Load a logo PNG, resize to square, return as numpy array."""
    img = Image.open(LOGOS_DIR / name).convert('RGBA')
    img = img.resize((size, size), Image.LANCZOS)
    return np.array(img)


def place_model_headers(ax, x_positions, y_pos, logos, names, colors, fontsize=7):
    """Place logo + model name above each bar position."""
    for xp, logo_file, name, color in zip(x_positions, logos, names, colors):
        # Place logo
        logo = load_logo_img(logo_file, size=80)
        im = OffsetImage(logo, zoom=0.28)
        ab = AnnotationBbox(im, (xp, y_pos), xycoords='data',
                            frameon=False, box_alignment=(0.5, 0.0))
        ax.add_artist(ab)
        # Place name below logo
        ax.text(xp, y_pos - 2, name, ha='center', va='top',
                fontsize=fontsize, fontweight='bold', color=color,
                linespacing=0.9)


def fig_benchmark_accuracy():
    """Grouped bar chart: accuracy/recall across 4 independent benchmarks.
    Layout: models as groups on x-axis, benchmarks as colored bars within each group.
    """
    benchmarks = ['NotInject\n(English)', 'Wildguard', 'Gandalf', 'InjecGuard']
    bench_colors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444']

    # Data: rows = models, cols = benchmarks
    data = {
        'VIGIL (Ours, 22M)':  [89.3, 97.8, 92.0, 71.5],
        'ProtectAI (200M)':   [62.5, 75.2, 100.0, 65.3],
        'Meta Guard (86M)':   [0.0,  6.7,  100.0, 43.8],
    }
    model_names = list(data.keys())
    model_logos = ['arvel-logo.png', 'protectai-logo.png', 'meta-logo.png']
    model_colors = [COLOR_OURS, COLOR_PAI, COLOR_META]

    n_models = len(model_names)
    n_bench = len(benchmarks)
    x = np.arange(n_models)
    width = 0.18
    offsets = np.arange(n_bench) - (n_bench - 1) / 2

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Draw bars — each model is a group, each benchmark is a bar within
    for j, (bench, color) in enumerate(zip(benchmarks, bench_colors)):
        vals = [data[m][j] for m in model_names]
        bars = ax.bar(x + offsets[j] * width, vals, width,
                       color=color, edgecolor='white', linewidth=0.5,
                       zorder=3, label=bench)
        # Value labels on bars
        for bar in bars:
            h = bar.get_height()
            if h > 3:
                ax.text(bar.get_x() + bar.get_width() / 2., h + 1.2,
                        f'{h:.0f}%', ha='center', va='bottom',
                        fontsize=7, fontweight='bold')

    # PIGuard reference line (only relevant for NotInject)
    ax.axhline(y=87.3, color=COLOR_PIGUARD, linewidth=1.5, linestyle='--',
               zorder=2, alpha=0.7)
    ax.text(n_models - 0.5, 88.5, 'PIGuard ref (87.3%)', fontsize=7,
            color=COLOR_PIGUARD, ha='right', fontweight='bold')

    # Place logos + model names below bars
    ax.set_xticks(x)
    ax.set_xticklabels(['' for _ in model_names])  # clear default labels

    for i, (name, logo, color) in enumerate(zip(model_names, model_logos, model_colors)):
        # Logo below x-axis
        logo_img = load_logo_img(logo, size=120)
        im = OffsetImage(logo_img, zoom=0.25)
        ab = AnnotationBbox(im, (i, 0), xycoords=('data', 'axes fraction'),
                            xybox=(0, -28), boxcoords='offset points',
                            frameon=False, box_alignment=(0.5, 1.0))
        ax.add_artist(ab)
        # Model name below logo
        ax.text(i, -0.02, name, ha='center', va='top',
                fontsize=9, fontweight='bold', color=color,
                transform=ax.get_xaxis_transform(),
                linespacing=0.85)

    ax.set_ylabel('Performance (%)', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 118)
    ax.set_title('Independent Benchmark Comparison', fontsize=14,
                 fontweight='bold', pad=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, zorder=0)

    # Benchmark legend (small, for the bar colors)
    metric_labels = ['Accuracy', 'Accuracy', 'Recall', 'Accuracy']
    legend_labels = [f'{b.replace(chr(10), " ")} ({m})' for b, m in zip(benchmarks, metric_labels)]
    ax.legend(loc='upper center', fontsize=7, ncol=4, framealpha=0.9,
              bbox_to_anchor=(0.5, 1.0), labels=legend_labels)

    plt.subplots_adjust(bottom=0.22)
    plt.savefig(FIGURES_DIR / 'classifier_benchmark_accuracy.png',
                dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'classifier_benchmark_accuracy.pdf',
                bbox_inches='tight')
    print("Saved: classifier_benchmark_accuracy")
    plt.close()


def fig_benchmark_fpr():
    """FPR comparison bar chart (lower is better).
    Models as groups on x-axis, benchmarks as colored bars.
    """
    benchmarks = ['NotInject (English)', 'Wildguard', 'InjecGuard']
    bench_colors = ['#3B82F6', '#10B981', '#EF4444']

    data = {
        'VIGIL (Ours, 22M)':  [10.7, 2.2, 12.5],
        'ProtectAI (200M)':   [37.5, 24.8, 30.2],
        'Meta Guard (86M)':   [100.0, 93.3, 84.4],
    }
    model_names = list(data.keys())
    model_logos = ['arvel-logo.png', 'protectai-logo.png', 'meta-logo.png']
    model_colors = [COLOR_OURS, COLOR_PAI, COLOR_META]

    n_models = len(model_names)
    n_bench = len(benchmarks)
    x = np.arange(n_models)
    width = 0.22
    offsets = np.arange(n_bench) - (n_bench - 1) / 2

    fig, ax = plt.subplots(figsize=(9, 5))

    for j, (bench, color) in enumerate(zip(benchmarks, bench_colors)):
        vals = [data[m][j] for m in model_names]
        bars = ax.bar(x + offsets[j] * width, vals, width,
                       color=color, edgecolor='white', linewidth=0.5,
                       zorder=3, label=bench)
        for bar in bars:
            h = bar.get_height()
            if h > 2:
                ax.text(bar.get_x() + bar.get_width() / 2., h + 1.2,
                        f'{h:.1f}%', ha='center', va='bottom',
                        fontsize=7, fontweight='bold')

    # PIGuard reference line
    ax.axhline(y=12.7, color=COLOR_PIGUARD, linewidth=1.5, linestyle='--',
               zorder=2, alpha=0.7)
    ax.text(n_models - 0.5, 14, 'PIGuard ref (12.7%)', fontsize=7,
            color=COLOR_PIGUARD, ha='right', fontweight='bold')

    # Logos + model names below x-axis
    ax.set_xticks(x)
    ax.set_xticklabels(['' for _ in model_names])

    for i, (name, logo, color) in enumerate(zip(model_names, model_logos, model_colors)):
        logo_img = load_logo_img(logo, size=120)
        im = OffsetImage(logo_img, zoom=0.25)
        ab = AnnotationBbox(im, (i, 0), xycoords=('data', 'axes fraction'),
                            xybox=(0, -28), boxcoords='offset points',
                            frameon=False, box_alignment=(0.5, 1.0))
        ax.add_artist(ab)
        ax.text(i, -0.02, name, ha='center', va='top',
                fontsize=9, fontweight='bold', color=color,
                transform=ax.get_xaxis_transform(),
                linespacing=0.85)

    ax.set_ylabel('False Positive Rate (%)', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 118)
    ax.set_title('False Positive Rate Comparison — Lower is Better',
                 fontsize=14, fontweight='bold', pad=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, zorder=0)

    # "Better" arrow
    ax.annotate('', xy=(-0.42, 5), xytext=(-0.42, 35),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.text(-0.52, 20, 'Better', rotation=90, fontsize=8,
            color='green', ha='right', fontweight='bold')

    # Benchmark legend
    ax.legend(loc='upper center', fontsize=7, ncol=3, framealpha=0.9,
              bbox_to_anchor=(0.5, 1.0))

    plt.subplots_adjust(bottom=0.22)
    plt.savefig(FIGURES_DIR / 'classifier_benchmark_fpr.png',
                dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'classifier_benchmark_fpr.pdf',
                bbox_inches='tight')
    print("Saved: classifier_benchmark_fpr")
    plt.close()


def fig_onnx_optimization():
    """ONNX optimization: latency and model size comparison."""
    models = ['PyTorch\nfp32', 'ONNX\nfp32', 'ONNX\nINT8']
    latency = [16.1, 6.1, 3.8]
    sizes = [271, 271, 83]
    colors = ['#94A3B8', '#3B82F6', '#2563EB']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

    # Latency
    bars_l = ax1.bar(models, latency, color=colors, edgecolor='white',
                     linewidth=0.5, zorder=3)
    for bar, val in zip(bars_l, latency):
        ax1.text(bar.get_x() + bar.get_width() / 2., val + 0.3,
                 f'{val}ms', ha='center', fontsize=9, fontweight='bold')
    ax1.set_ylabel('Latency (ms)', fontsize=10, fontweight='bold')
    ax1.set_title('Single Text Inference (CPU)', fontsize=11, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_ylim(0, 20)
    ax1.grid(axis='y', alpha=0.3, zorder=0)

    # Speedup annotations
    ax1.annotate('2.6x faster', xy=(1, 6.1), fontsize=8, color='green',
                 fontweight='bold', ha='center', va='bottom',
                 xytext=(1, 9), arrowprops=dict(arrowstyle='->', color='green', lw=1.2))
    ax1.annotate('4.2x faster', xy=(2, 3.8), fontsize=8, color='green',
                 fontweight='bold', ha='center', va='bottom',
                 xytext=(2, 7), arrowprops=dict(arrowstyle='->', color='green', lw=1.2))

    # Model size
    bars_s = ax2.bar(models, sizes, color=colors, edgecolor='white',
                     linewidth=0.5, zorder=3)
    for bar, val in zip(bars_s, sizes):
        ax2.text(bar.get_x() + bar.get_width() / 2., val + 5,
                 f'{val} MB', ha='center', fontsize=9, fontweight='bold')
    ax2.set_ylabel('Model Size (MB)', fontsize=10, fontweight='bold')
    ax2.set_title('Model Size on Disk', fontsize=11, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_ylim(0, 320)
    ax2.grid(axis='y', alpha=0.3, zorder=0)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'classifier_onnx_optimization.png',
                dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'classifier_onnx_optimization.pdf',
                bbox_inches='tight')
    print("Saved: classifier_onnx_optimization")
    plt.close()


if __name__ == '__main__':
    fig_benchmark_accuracy()
    fig_benchmark_fpr()
    fig_onnx_optimization()
    print("\nAll classifier figures generated!")
