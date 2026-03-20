"""Prepare 1-class cleaned dataset from 5-class crawled data.

Merges all 5 classes into class 0 (ui_element), removes duplicate
and tiny boxes, and creates a new dataset directory.
"""

import os
import subprocess
from pathlib import Path

DATA_ROOT = Path("data")
MIN_AREA = 0.0001  # Minimum box area (0.01% of image) — smaller = noise


def clean_labels(src_dir: Path, dst_dir: Path):
    """Merge classes to 0, remove duplicates and tiny boxes."""
    dst_dir.mkdir(parents=True, exist_ok=True)

    total_files = 0
    total_boxes_in = 0
    total_boxes_out = 0
    dupes_removed = 0
    tiny_removed = 0

    for src_file in src_dir.iterdir():
        if not src_file.suffix == ".txt":
            continue

        total_files += 1
        lines = src_file.read_text().strip().split("\n")
        if not lines or lines == [""]:
            # Write empty file
            (dst_dir / src_file.name).write_text("")
            continue

        seen = set()
        clean_lines = []

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            total_boxes_in += 1

            _, xc, yc, w, h = parts
            xc, yc, w, h = float(xc), float(yc), float(w), float(h)

            # Remove tiny boxes
            if w * h < MIN_AREA:
                tiny_removed += 1
                continue

            # Deduplicate (round to 3 decimal places)
            key = f"{round(xc, 3)}_{round(yc, 3)}_{round(w, 3)}_{round(h, 3)}"
            if key in seen:
                dupes_removed += 1
                continue
            seen.add(key)

            # Merge to class 0
            clean_lines.append(f"0 {xc} {yc} {w} {h}")
            total_boxes_out += 1

        (dst_dir / src_file.name).write_text("\n".join(clean_lines) + "\n" if clean_lines else "")

        if total_files % 50000 == 0:
            print(f"  Processed {total_files} files...")

    return total_files, total_boxes_in, total_boxes_out, dupes_removed, tiny_removed


def make_junction(src: Path, dest: Path):
    """Create Windows directory junction."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        try:
            os.rmdir(str(dest))
        except OSError:
            pass
    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(dest), str(src)],
        check=True, capture_output=True,
    )


def main():
    # Clean crawled labels
    print("Cleaning crawled labels (258K files)...")
    stats = clean_labels(
        DATA_ROOT / "crawled" / "labels",
        DATA_ROOT / "crawled_1cls" / "labels",
    )
    print(f"  Files: {stats[0]}")
    print(f"  Boxes in: {stats[1]}")
    print(f"  Boxes out: {stats[2]}")
    print(f"  Duplicates removed: {stats[3]} ({stats[3]/stats[1]*100:.1f}%)")
    print(f"  Tiny removed: {stats[4]} ({stats[4]/stats[1]*100:.1f}%)")

    # Junction for images (no need to copy 48GB)
    print("\nLinking crawled images...")
    make_junction(
        DATA_ROOT / "crawled" / "images",
        DATA_ROOT / "crawled_1cls" / "images",
    )

    # Clean val/test labels
    for split in ["val", "test"]:
        print(f"\nCleaning {split} labels...")
        stats = clean_labels(
            DATA_ROOT / "raw" / split / "labels",
            DATA_ROOT / "raw_1cls" / split / "labels",
        )
        print(f"  Files: {stats[0]}, Boxes: {stats[1]} -> {stats[2]}")

        # Junction for images
        make_junction(
            DATA_ROOT / "raw" / split / "images",
            DATA_ROOT / "raw_1cls" / split / "images",
        )

    # Write dataset.yaml
    yaml_content = """path: C:/Users/Alden Bernstein/desktop/COGS181-Final/data
train: crawled_1cls/images
val: raw_1cls/val/images
test: raw_1cls/test/images
nc: 1
names: [ui_element]
"""
    yaml_path = DATA_ROOT / "crawled_1cls" / "dataset.yaml"
    yaml_path.write_text(yaml_content)
    print(f"\nDataset yaml: {yaml_path}")
    print("Done!")


if __name__ == "__main__":
    main()
