"""Remap 16-class YOLO labels to 5 balanced classes.

Mapping:
  0 interactive: link(0), button(1), menu_item(11), clickable(12)
  1 form_input:  input(2), select(3), textarea(4), dropdown(8)
  2 form_control: checkbox(6), radio(7), slider(9), toggle(10)
  3 text_label:  label(5), text(15)
  4 media:       icon(13), image(14)

Usage:
    python remap_labels.py
"""

import os
import shutil
from pathlib import Path

REMAP = {
    0: 0, 1: 0, 11: 0, 12: 0,   # interactive
    2: 1, 3: 1, 4: 1, 8: 1,      # form_input
    6: 2, 7: 2, 9: 2, 10: 2,     # form_control
    5: 3, 15: 3,                   # text_label
    13: 4, 14: 4,                  # media
}

BASE = Path("data")

# (source_labels, dest_labels, source_images, dest_images)
DIRS = [
    (BASE / "crawled/labels", BASE / "crawled_5cls/labels",
     BASE / "crawled/images", BASE / "crawled_5cls/images"),
    (BASE / "raw/train/labels", BASE / "raw_5cls/train/labels",
     BASE / "raw/train/images", BASE / "raw_5cls/train/images"),
    (BASE / "raw/val/labels", BASE / "raw_5cls/val/labels",
     BASE / "raw/val/images", BASE / "raw_5cls/val/images"),
    (BASE / "raw/test/labels", BASE / "raw_5cls/test/labels",
     BASE / "raw/test/images", BASE / "raw_5cls/test/images"),
]


def remap_file(src, dst):
    """Remap class IDs in a single label file."""
    lines = []
    with open(src) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            old_cls = int(parts[0])
            if old_cls in REMAP:
                parts[0] = str(REMAP[old_cls])
                lines.append(" ".join(parts))
    with open(dst, "w") as f:
        f.write("\n".join(lines) + "\n" if lines else "")


def main():
    for src_labels, dst_labels, src_images, dst_images in DIRS:
        if not src_labels.exists():
            print(f"  Skipping {src_labels} (not found)")
            continue

        # Create dest label dir
        dst_labels.mkdir(parents=True, exist_ok=True)

        # Symlink images dir (no copy needed)
        if not dst_images.exists():
            dst_images.parent.mkdir(parents=True, exist_ok=True)
            # On Windows, use junction instead of symlink (no admin needed)
            abs_src = str(src_images.resolve())
            abs_dst = str(dst_images.resolve())
            if os.name == "nt":
                os.system(f'mklink /J "{abs_dst}" "{abs_src}"')
            else:
                os.symlink(abs_src, abs_dst)
            print(f"  Linked {dst_images} -> {src_images}")

        # Remap labels
        files = [f for f in os.listdir(src_labels) if f.endswith(".txt")]
        print(f"  Remapping {len(files)} labels: {src_labels} -> {dst_labels}")
        for i, fname in enumerate(files):
            remap_file(src_labels / fname, dst_labels / fname)
            if (i + 1) % 10000 == 0:
                print(f"    {i+1}/{len(files)}")

        print(f"  Done: {dst_labels}")

    print("\nAll remapping complete.")


if __name__ == "__main__":
    main()
