# YOLOv8 for Web UI Element Detection: A CNN Ablation Study

**COGS 181 Final Project** -- Alden Bernstein

## Overview

We apply YOLOv8 (a CNN-based object detector) to detecting UI elements in web page screenshots. We crawl ~260K screenshots from the Tranco Top 1M, then conduct **14 controlled ablation experiments** covering architectures, layers, optimizers, pooling, and activation functions.

The main finding: **data preparation matters more than hyperparameters.** Our iterative data pipeline (16 classes → 5 classes → 1 class with deduplication and diverse sampling) improves mAP50 from 0.136 to **0.323** — a 2.4x gain using the same model. This independently confirms OmniParser's single-class approach on our own data.

We also train a DeBERTa text classifier as a practical extension.

Development was assisted by Claude Code.

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

### 4. Train text classifier (practical extension)

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
├── inference.py                   # Single-image YOLO inference
├── practical_application.py       # Full detection + classification pipeline
├── optimize_model.py              # ONNX export + INT8 quantization
├── requirements.txt
├── models/
│   ├── custom_yolo.yaml           # Custom lightweight architecture (0.71 MB)
│   ├── custom_yolo_deep.yaml      # Doubled depth variant
│   ├── custom_yolo_relu.yaml      # ReLU activation variant
│   └── custom_yolo_avgpool.yaml   # Average pooling variant
├── report/icml2023/               # LaTeX report (ICML format)
├── results_yolo_5cls/             # 14 ablation experiment results
└── results_yolo_1cls/             # 1-class experiment results
```

## Key Results

| Setup | mAP50 | Recall | Training Images |
|---|---|---|---|
| 5-class baseline (YOLOv8n) | 0.002 | 0.005 | 22K |
| 5-class best combo (YOLOv8s) | 0.136 | 0.169 | 22K |
| **1-class diverse (YOLOv8n)** | **0.323** | **0.426** | **47K** |

Data cleaning and class consolidation gave a 2.4x improvement over the best hyperparameter-tuned model.
