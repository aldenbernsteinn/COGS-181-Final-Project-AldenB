"""YOLO-based UI element detection inference.

Detects UI elements (buttons, inputs, links, icons, etc.) in screenshots
using a trained YOLOv8 model.

Usage:
    python inference.py --image screenshot.png
    python inference.py --image screenshot.png --model results_yolo/arch_yolov8n/weights/best.pt
    python inference.py --image_dir test_screenshots/ --save_dir output/
"""

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# 16 UI element classes from dataset.yaml
CLASS_NAMES = [
    "link", "button", "input", "select", "textarea", "label",
    "checkbox", "radio", "dropdown", "slider", "toggle", "menu_item",
    "clickable", "icon", "image", "text",
]

# Colors for each class (RGB)
CLASS_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 165, 0),
    (128, 0, 128), (0, 128, 128), (255, 192, 203), (165, 42, 42),
    (0, 255, 255), (255, 255, 0), (75, 0, 130), (255, 127, 80),
    (0, 128, 0), (70, 130, 180), (220, 20, 60), (128, 128, 0),
]

DEFAULT_MODEL = "results_yolo/best_combo/weights/best.pt"


def detect(model, image_path, conf=0.25, imgsz=640):
    """Run detection on a single image. Returns list of detections."""
    results = model.predict(
        str(image_path), conf=conf, imgsz=imgsz, verbose=False,
    )
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "class_id": int(box.cls[0]),
                "class_name": CLASS_NAMES[int(box.cls[0])],
                "confidence": round(float(box.conf[0]), 4),
                "bbox": [round(x1), round(y1), round(x2), round(y2)],
            })
    return detections


def draw_detections(image_path, detections, output_path=None):
    """Draw bounding boxes and labels on the image."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = CLASS_COLORS[det["class_id"] % len(CLASS_COLORS)]
        label = f"{det['class_name']} {det['confidence']:.2f}"

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle(
            [text_bbox[0] - 1, text_bbox[1] - 1, text_bbox[2] + 1, text_bbox[3] + 1],
            fill=color,
        )
        draw.text((x1, y1), label, fill="white", font=font)

    if output_path:
        img.save(output_path)
    return img


def print_summary(detections):
    """Print detection summary."""
    print(f"\nDetected {len(detections)} elements:")
    class_counts = {}
    for det in detections:
        name = det["class_name"]
        class_counts[name] = class_counts.get(name, 0) + 1
    for name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count}")


def benchmark(model, imgsz=640, n_runs=20):
    """Benchmark inference speed."""
    dummy = torch.rand(1, 3, imgsz, imgsz)

    # Warmup
    for _ in range(5):
        model.predict(dummy, verbose=False)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict(dummy, verbose=False)
        times.append((time.perf_counter() - t0) * 1000)

    avg = sum(times) / len(times)
    print(f"\nInference speed: {avg:.1f} ms/image (CPU)")
    return avg


def main():
    parser = argparse.ArgumentParser(description="YOLO UI Element Detection")
    parser.add_argument("--image", type=str, help="Path to single image")
    parser.add_argument("--image_dir", type=str, help="Path to directory of images")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Path to YOLO weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--save_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--benchmark", action="store_true", help="Run speed benchmark")
    parser.add_argument("--json", action="store_true", help="Output detections as JSON")
    args = parser.parse_args()

    model = YOLO(args.model)

    if args.benchmark:
        benchmark(model, args.imgsz)
        return

    save_dir = Path(args.save_dir)

    if args.image:
        detections = detect(model, args.image, args.conf, args.imgsz)
        if args.json:
            print(json.dumps(detections, indent=2))
        else:
            print_summary(detections)
            save_dir.mkdir(parents=True, exist_ok=True)
            out_path = save_dir / Path(args.image).name
            draw_detections(args.image, detections, out_path)
            print(f"Saved to {out_path}")

    elif args.image_dir:
        image_dir = Path(args.image_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        paths = sorted(
            p for p in image_dir.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp")
        )
        print(f"Processing {len(paths)} images...")
        all_detections = {}
        for p in paths:
            dets = detect(model, p, args.conf, args.imgsz)
            all_detections[p.name] = dets
            draw_detections(p, dets, save_dir / p.name)
            print(f"  {p.name}: {len(dets)} elements")

        with open(save_dir / "detections.json", "w") as f:
            json.dump(all_detections, f, indent=2)
        print(f"\nProcessed {len(paths)} images -> {save_dir}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
