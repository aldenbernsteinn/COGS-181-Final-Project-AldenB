# YOLOv8 for Web UI Element Detection: A CNN Ablation Study

**COGS 181 Final Project** -- Alden Bernstein

## Overview

We apply YOLOv8 (a CNN-based object detector) to detecting UI elements in web page screenshots. We crawl ~260K screenshots from the Tranco Top 1M, then conduct **14 controlled ablation experiments** covering architectures, layers, optimizers, pooling, and activation functions.

The ablation's per-class analysis revealed that 3 of 5 classes were undetectable regardless of configuration. Following this finding, we restructured the task into single-class detection with deduplication and diverse sampling, improving mAP50 from 0.136 to **0.323** (2.4x) — supporting OmniParser's single-class approach on independent data.

We also train a DeBERTa text classifier as a bonus exploration for prompt injection detection.

Development of the report and scripts was assisted by Claude Code.

## Setup

```bash
pip install -r requirements.txt
playwright install chromium  # Only needed for data crawling
```

## Reproducing Results

### 1. Crawl training data

```bash
python generate_training_data.py --num-pages 1000 --workers 4
```

Crawls websites from the Tranco Top 1M, takes screenshots, extracts UI element bounding boxes from the DOM. Full dataset (~260K images) needs `--num-pages 70000`.

### 2. Prepare data (16 → 5 → 1 class pipeline)

```bash
python remap_labels.py              # 16 fine-grained classes → 5 groups
python prepare_1cls_data.py         # 5 classes → 1 class (ui_element), dedup, remove tiny boxes
python prepare_diverse_subset.py    # Select 1 image per domain (~47K diverse images)
```

### 3. Train YOLO (14 ablation experiments)

```bash
python train_yolo.py                    # Run all 14 experiments (5-class)
python train_yolo.py --experiment arch  # Run specific group
python train_yolo.py --summary          # Print results
```

### 4. Train text classifier (bonus)

```bash
python train_injection_classifier.py              # Train DeBERTa classifier
python train_injection_classifier.py --benchmark  # Benchmark evaluation
```

### 5. Generate figures

```bash
python generate_figures.py
```

## Project Structure

```
.
├── train_yolo.py                  # YOLO experiment runner (14 ablations)
├── remap_labels.py                # 16-class → 5-class label remapping
├── prepare_1cls_data.py           # 5-class → 1-class consolidation + dedup
├── prepare_diverse_subset.py      # 1-per-domain diverse sampling
├── generate_training_data.py      # Playwright web crawler
├── train_injection_classifier.py  # DeBERTa text classifier
├── generate_figures.py            # Report figures
├── generate_classifier_figures.py # Classifier benchmark figures
├── inference.py                   # Single-image YOLO inference
├── practical_application.py       # Full detection + classification pipeline
├── optimize_model.py              # ONNX export + INT8 quantization
├── requirements.txt
├── models/
│   ├── custom_yolo.yaml           # Custom lightweight architecture (0.71 MB)
│   ├── custom_yolo_deep.yaml      # Doubled depth variant
│   ├── custom_yolo_relu.yaml      # ReLU activation variant
│   ├── custom_yolo_avgpool.yaml   # Average pooling variant
│   └── sppf_avg.py                # AvgPool2d SPPF module
├── report/icml2023/               # LaTeX report (ICML format)
├── results_yolo_5cls/             # 14 ablation results (JSON + per-epoch CSV)
├── results_yolo_1cls/             # 1-class experiment results (JSON + CSV)
├── results_classifier/            # 17 classifier experiment results (JSON)
└── optimized_models/              # ONNX quantization benchmark results
```

## Key Results

### Hyperparameter Ablation (14 experiments, 5-class)

| Category | Finding | mAP50 |
|---|---|---|
| (a) Architecture | YOLOv8s (11.2M) best; YOLOv8m (25.9M) no further gain | 0.134 |
| (b) Layers | Doubling depth does not improve over baseline | 0.107 |
| (c) Optimizer | AdamW slightly better recall than SGD | 0.110 |
| (d) Pooling | Max pooling slightly outperforms average pooling | 0.108 vs 0.106 |
| (e) Activation | SiLU slightly outperforms ReLU | 0.108 vs 0.103 |

### Final Model (1-class, restructured task)

| Model | mAP50 | Precision | Recall | CPU (ms) |
|---|---|---|---|---|
| YOLOv8n, 1-class diverse (47K images) | **0.323** | 0.427 | 0.426 | 19.8 |

Architecture produced the largest single-factor gains among hyperparameters. The ablation's per-class analysis then revealed that the initial 16-class formulation was too granular, leading to the 1-class restructuring (2.4x improvement).

All experiment results are included as JSON and CSV files in the `results_*/` directories, so paper numbers and training curves can be verified without retraining.
