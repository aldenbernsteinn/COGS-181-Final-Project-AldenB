"""Train YOLOv8 models for UI element detection with comprehensive experiments.

Runs a grid of experiments varying architecture, image size, learning rate,
epochs, augmentation, and pretrained weights. Compares against OmniParser
baseline. Saves results to results_yolo/<experiment_name>/.

Usage:
    python train_yolo.py                    # Run all experiments
    python train_yolo.py --experiment arch  # Run specific experiment group
    python train_yolo.py --list             # List all experiments
    python train_yolo.py --eval-omniparser  # Evaluate OmniParser baseline
"""

import argparse
import gc
import json
import os
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from ultralytics import YOLO

# ─── Constants ────────────────────────────────────────────────────────────────

RESULTS_DIR = Path("results_yolo_5cls")
DATA_YAML = Path("data/crawled_5cls/dataset.yaml")
DEVICE = "0" if torch.cuda.is_available() else "cpu"
DEFAULT_EPOCHS = 25
DEFAULT_IMGSZ = 640
DEFAULT_LR = 0.01
DEFAULT_BATCH = 16 if torch.cuda.is_available() else 8
DEFAULT_FRACTION = 0.085

# ─── Experiment definitions ──────────────────────────────────────────────────


def _base(data_yaml, **overrides):
    """Baseline config with overrides. Ensures no duplicate experiments."""
    config = {
        "model": "yolov8n.pt",
        "data": data_yaml,
        "epochs": DEFAULT_EPOCHS,
        "imgsz": DEFAULT_IMGSZ,
        "lr0": DEFAULT_LR,
        "batch": DEFAULT_BATCH,
        "fraction": DEFAULT_FRACTION,
        "optimizer": "auto",
        "pretrained": True,
        "mosaic": 1.0,
    }
    config.update(overrides)
    return config


def get_experiments(data_yaml=None):
    """Define all experiments. Each varies exactly ONE dimension from baseline."""
    if data_yaml is None:
        data_yaml = str(DATA_YAML)

    experiments = {}

    # ── Baseline: YOLOv8n, 640, auto/SGD, lr=0.01, 50ep, mosaic, pretrained ──
    experiments["baseline"] = {
        **_base(data_yaml), "group": "baseline",
    }

    # ── Architecture comparison (Task.md items a, b) ──
    experiments["arch_yolov8s"] = {
        **_base(data_yaml, model="yolov8s.pt"), "group": "arch",
    }
    experiments["arch_yolov8m"] = {
        **_base(data_yaml, model="yolov8m.pt"), "group": "arch",
    }
    custom_yaml = Path("models/custom_yolo.yaml")
    if custom_yaml.exists():
        experiments["arch_custom_cpu"] = {
            **_base(data_yaml, model=str(custom_yaml), pretrained=False),
            "group": "arch",
        }

    # ── Optimizer comparison (Task.md item c) ──
    experiments["optimizer_adamw"] = {
        **_base(data_yaml, optimizer="AdamW"), "group": "optimizer",
    }

    # ── Learning rate (Task.md item c) ──
    experiments["lr_0.001"] = {
        **_base(data_yaml, lr0=0.001), "group": "lr",
    }
    experiments["lr_0.005"] = {
        **_base(data_yaml, lr0=0.005), "group": "lr",
    }

    # ── Image resolution ──
    experiments["imgsz_1280"] = {
        **_base(data_yaml, imgsz=1280, batch=8),
        "group": "imgsz",
    }

    # ── Augmentation ──
    experiments["aug_no_mosaic"] = {
        **_base(data_yaml, mosaic=0.0), "group": "augmentation",
    }

    # ── Initialization (pretrained vs scratch) ──
    experiments["init_scratch"] = {
        **_base(data_yaml, model="yolov8n.yaml", pretrained=False),
        "group": "init",
    }

    # ── Layers / depth comparison (Task.md item b) ──
    custom_deep = Path("models/custom_yolo_deep.yaml")
    if custom_deep.exists():
        experiments["layers_deep"] = {
            **_base(data_yaml, model=str(custom_deep), pretrained=False),
            "group": "layers",
        }

    # ── Activation comparison (Task.md item e) ──
    custom_relu = Path("models/custom_yolo_relu.yaml")
    if custom_relu.exists():
        experiments["activation_relu"] = {
            **_base(data_yaml, model=str(custom_relu), pretrained=False),
            "group": "activation",
        }

    # ── Pooling comparison (Task.md item d) ──
    custom_avgpool = Path("models/custom_yolo_avgpool.yaml")
    if custom_avgpool.exists():
        experiments["pooling_avg"] = {
            **_base(data_yaml, model=str(custom_avgpool), pretrained=False),
            "group": "pooling",
            "_patch_avgpool": True,
        }

    # ── Best combo: combine best setting from each axis (runs last) ──
    experiments["best_combo"] = {
        **_base(data_yaml), "group": "best_combo", "_dynamic": True,
    }

    return experiments


def resolve_best_combo(experiments):
    """Build best_combo config dynamically from completed experiment results."""
    # For each axis, find which setting won (highest mAP50)
    axis_groups = {
        "arch": {"key": "model", "baseline_val": "yolov8n.pt"},
        "optimizer": {"key": "optimizer", "baseline_val": "auto"},
        "lr": {"key": "lr0", "baseline_val": DEFAULT_LR},
        "imgsz": {"key": "imgsz", "baseline_val": DEFAULT_IMGSZ},
        "augmentation": {"key": "mosaic", "baseline_val": 1.0},
        "init": {"key": "pretrained", "baseline_val": True},
    }

    best_overrides = {}
    for group, info in axis_groups.items():
        # Collect all completed experiments in this group + baseline
        candidates = []
        for name, config in experiments.items():
            if config.get("group") in (group, "baseline"):
                result = load_results(name)
                if result:
                    candidates.append((name, config, result["metrics"]["mAP50"]))

        if not candidates:
            continue

        # Pick the winner
        winner_name, winner_config, winner_map = max(candidates, key=lambda x: x[2])
        val = winner_config.get(info["key"], info["baseline_val"])
        if val != info["baseline_val"]:
            best_overrides[info["key"]] = val
            print(f"  best_combo: {info['key']}={val} (from {winner_name}, mAP50={winner_map:.4f})")

    # Handle special cases: if best arch needs pretrained=False or different batch
    if best_overrides.get("model", "").endswith(".yaml"):
        best_overrides["pretrained"] = False
    if best_overrides.get("imgsz", DEFAULT_IMGSZ) == 1280:
        best_overrides["batch"] = 8

    combo_config = {**experiments["best_combo"]}
    del combo_config["_dynamic"]
    combo_config.update(best_overrides)

    if not best_overrides:
        print("  best_combo: baseline won every axis — skipping")
        return None

    print(f"  best_combo final config: {best_overrides}")
    return combo_config


# ─── Model patches ───────────────────────────────────────────────────────────


def reset_conv_activation():
    """Reset Conv.default_act to SiLU (Ultralytics default).
    Needed because loading a YAML with `activation: nn.ReLU()` sets this
    class variable globally, contaminating subsequent models.
    """
    from ultralytics.nn.modules.conv import Conv
    Conv.default_act = nn.SiLU()


_sppf_original_init = None


def enable_sppf_avgpool():
    """Monkey-patch SPPF.__init__ to use AvgPool2d instead of MaxPool2d."""
    from ultralytics.nn.modules.block import SPPF
    global _sppf_original_init
    _sppf_original_init = SPPF.__init__

    def patched_init(self, c1, c2, k=5):
        _sppf_original_init(self, c1, c2, k)
        self.m = nn.AvgPool2d(kernel_size=k, stride=1, padding=k // 2)

    SPPF.__init__ = patched_init


def disable_sppf_avgpool():
    """Restore original SPPF.__init__."""
    from ultralytics.nn.modules.block import SPPF
    global _sppf_original_init
    if _sppf_original_init is not None:
        SPPF.__init__ = _sppf_original_init
        _sppf_original_init = None


# ─── Training ────────────────────────────────────────────────────────────────


def train_experiment(name, config):
    """Train a single experiment and save results. Auto-reduces batch on OOM."""
    exp_dir = RESULTS_DIR / name
    if (exp_dir / "results.json").exists():
        print(f"  Skipping {name} (already done)")
        return load_results(name)

    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"Config: {json.dumps({k:v for k,v in config.items() if k != 'group'}, indent=2)}")
    print(f"Device: {DEVICE}")
    print(f"{'='*60}")

    start = time.time()

    # OOM-safe training: retry with halved batch on CUDA OOM
    batch = config["batch"]
    model = None
    results = None
    while batch >= 2:
        try:
            torch.cuda.empty_cache()
            gc.collect()
            # Apply class-level patches before model construction
            if config.get("_patch_avgpool"):
                enable_sppf_avgpool()
            model = YOLO(config["model"])
            results = model.train(
                data=config["data"],
                epochs=config["epochs"],
                imgsz=config["imgsz"],
                lr0=config["lr0"],
                batch=batch,
                optimizer=config.get("optimizer", "auto"),
                fraction=config.get("fraction", 0.3),
                device=DEVICE,
                project=str(RESULTS_DIR),
                name=name,
                exist_ok=True,
                mosaic=config["mosaic"],
                verbose=True,
                save=True,
                plots=True,
                workers=8,
                cache=False,
            )
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n  OOM at batch={batch}, retrying with batch={batch // 2}")
                del model
                model = None
                torch.cuda.empty_cache()
                gc.collect()
                batch //= 2
                # Clean partial results so Ultralytics doesn't get confused
                shutil.rmtree(exp_dir, ignore_errors=True)
                exp_dir.mkdir(parents=True, exist_ok=True)
            else:
                raise

    if results is None:
        raise RuntimeError(f"OOM even at batch=2 for {name}")

    train_time = time.time() - start

    # Evaluate on test set
    metrics = model.val(
        data=config["data"],
        split="test",
        device=DEVICE,
        imgsz=config["imgsz"],
    )

    # Benchmark inference speed
    speed = benchmark_speed(model, config["imgsz"])

    # Collect results
    result_data = {
        "experiment": name,
        "config": {k: v for k, v in config.items() if k != "group" and not k.startswith("_")},
        "actual_batch": batch,
        "group": config["group"],
        "train_time_seconds": train_time,
        "metrics": {
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "per_class_ap50": {
                name: float(ap)
                for name, ap in zip(metrics.names.values(), metrics.box.ap50)
            } if hasattr(metrics.box, 'ap50') else {},
        },
        "speed": speed,
        "model_size_mb": get_model_size(exp_dir),
    }

    # Save results
    with open(exp_dir / "results.json", "w") as f:
        json.dump(result_data, f, indent=2)

    # Copy best weights
    best_pt = exp_dir / "weights" / "best.pt"
    if best_pt.exists():
        shutil.copy(best_pt, exp_dir / "best.pt")

    print(f"\n  mAP50: {result_data['metrics']['mAP50']:.4f}")
    print(f"  mAP50-95: {result_data['metrics']['mAP50_95']:.4f}")
    print(f"  Precision: {result_data['metrics']['precision']:.4f}")
    print(f"  Recall: {result_data['metrics']['recall']:.4f}")
    print(f"  Train time: {train_time/60:.1f} min")
    print(f"  GPU speed: {speed.get('gpu_ms', 'N/A')} ms/img")
    print(f"  CPU speed: {speed.get('cpu_ms', 'N/A')} ms/img")

    # Cleanup after experiment — reset class-level patches
    del model
    reset_conv_activation()
    disable_sppf_avgpool()
    torch.cuda.empty_cache()
    gc.collect()

    return result_data


def benchmark_speed(model, imgsz):
    """Benchmark inference speed on GPU and CPU."""
    speed = {}

    # Create dummy input
    dummy = torch.rand(1, 3, imgsz, imgsz)

    # GPU benchmark
    if torch.cuda.is_available():
        model_gpu = model.to("cuda")
        # Warmup
        for _ in range(5):
            model_gpu.predict(dummy.cuda(), verbose=False)
        torch.cuda.synchronize()

        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            model_gpu.predict(dummy.cuda(), verbose=False)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        speed["gpu_ms"] = round(sum(times) / len(times), 2)

    # CPU benchmark
    model_cpu = model.to("cpu")
    # Warmup
    for _ in range(3):
        model_cpu.predict(dummy, verbose=False)

    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        model_cpu.predict(dummy, verbose=False)
        times.append((time.perf_counter() - t0) * 1000)
    speed["cpu_ms"] = round(sum(times) / len(times), 2)

    return speed


def get_model_size(exp_dir):
    """Get model file size in MB."""
    best_pt = exp_dir / "weights" / "best.pt"
    if best_pt.exists():
        return round(best_pt.stat().st_size / (1024 * 1024), 2)
    return None


def load_results(name):
    """Load saved results for an experiment."""
    path = RESULTS_DIR / name / "results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ─── OmniParser evaluation ──────────────────────────────────────────────────


def eval_omniparser():
    """Download and evaluate OmniParser's pretrained icon detection model."""
    print("Evaluating OmniParser v2 baseline...")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("  pip install huggingface_hub")
        return None

    # Download OmniParser icon detection model
    model_path = hf_hub_download(
        repo_id="microsoft/OmniParser-v2.0",
        filename="icon_detect/best.pt",
        cache_dir=".cache/omniparser",
    )

    model = YOLO(model_path)

    # Evaluate on our test set
    data_yaml = str(DATA_YAML)
    metrics = model.val(data=data_yaml, split="test", device=DEVICE)

    speed = benchmark_speed(model, 640)

    result = {
        "experiment": "omniparser_baseline",
        "config": {"model": "OmniParser-v2.0/icon_detect/best.pt"},
        "group": "baseline",
        "metrics": {
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        },
        "speed": speed,
    }

    # Save
    omni_dir = RESULTS_DIR / "omniparser_baseline"
    omni_dir.mkdir(parents=True, exist_ok=True)
    with open(omni_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  OmniParser mAP50: {result['metrics']['mAP50']:.4f}")
    print(f"  OmniParser Recall: {result['metrics']['recall']:.4f}")
    return result


# ─── Summary ─────────────────────────────────────────────────────────────────


def print_summary():
    """Print summary of all experiment results."""
    print(f"\n{'='*80}")
    print(f"EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"{'Experiment':<30} {'mAP50':>8} {'mAP50-95':>10} {'Prec':>8} {'Recall':>8} "
          f"{'GPU(ms)':>8} {'CPU(ms)':>8} {'Size(MB)':>10}")
    print("-" * 100)

    all_results = []
    for exp_dir in sorted(RESULTS_DIR.iterdir()):
        result_file = exp_dir / "results.json"
        if result_file.exists():
            with open(result_file) as f:
                result = json.load(f)
                all_results.append(result)
                m = result["metrics"]
                s = result.get("speed", {})
                sz = result.get("model_size_mb", "")
                print(f"{result['experiment']:<30} "
                      f"{m['mAP50']:>8.4f} {m['mAP50_95']:>10.4f} "
                      f"{m['precision']:>8.4f} {m['recall']:>8.4f} "
                      f"{s.get('gpu_ms', 'N/A'):>8} {s.get('cpu_ms', 'N/A'):>8} "
                      f"{sz if sz else 'N/A':>10}")

    # Save all results summary
    if all_results:
        with open(RESULTS_DIR / "all_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved summary to {RESULTS_DIR / 'all_results.json'}")

    # Best model
    if all_results:
        best = max(all_results, key=lambda x: x["metrics"]["mAP50"])
        print(f"\nBest model: {best['experiment']} (mAP50={best['metrics']['mAP50']:.4f})")


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 for UI element detection")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Run specific experiment group (baseline, arch, optimizer, lr, imgsz, augmentation, init, best)")
    parser.add_argument("--list", action="store_true", help="List all experiments")
    parser.add_argument("--eval-omniparser", action="store_true", help="Evaluate OmniParser baseline")
    parser.add_argument("--summary", action="store_true", help="Print results summary")
    args = parser.parse_args()

    experiments = get_experiments()

    if args.list:
        print("Available experiments:")
        for name, config in experiments.items():
            print(f"  {name} [{config['group']}]")
        return

    if args.summary:
        print_summary()
        return

    if args.eval_omniparser:
        eval_omniparser()
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Filter by group if specified
    if args.experiment:
        experiments = {k: v for k, v in experiments.items() if v["group"] == args.experiment}
        if not experiments:
            print(f"No experiments found for group: {args.experiment}")
            print("Available groups: baseline, arch, optimizer, lr, imgsz, augmentation, init, best")
            return

    print(f"Running {len(experiments)} experiments...")
    print(f"Device: {DEVICE}")
    print(f"Results dir: {RESULTS_DIR}")

    for name, config in experiments.items():
        try:
            # Resolve best_combo dynamically after all other experiments
            if config.get("_dynamic"):
                print(f"\nResolving best_combo from completed results...")
                combo = resolve_best_combo(experiments)
                if combo is None:
                    continue
                config = combo
            train_experiment(name, config)
        except Exception as e:
            print(f"\n  ERROR in {name}: {e}")
            # Save error
            err_dir = RESULTS_DIR / name
            err_dir.mkdir(parents=True, exist_ok=True)
            with open(err_dir / "error.txt", "w") as f:
                f.write(str(e))
        finally:
            # Clear GPU memory between experiments
            torch.cuda.empty_cache()
            gc.collect()

    print_summary()


if __name__ == "__main__":
    main()
