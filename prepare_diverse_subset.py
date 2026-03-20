"""Create a diverse 1-per-domain subset of the crawled dataset.

Selects 1 image per domain (preferring 'top' section) and symlinks
images + copies cleaned 1-class labels.
"""

import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

DATA_ROOT = Path("data")
SRC_IMAGES = DATA_ROOT / "crawled" / "images"
SRC_LABELS = DATA_ROOT / "crawled_1cls" / "labels"
DST_DIR = DATA_ROOT / "crawled_1cls_diverse"
DST_IMAGES = DST_DIR / "images"
DST_LABELS = DST_DIR / "labels"

# Section priority: prefer top (main page view)
SECTION_PRIORITY = {"top": 0, "mid": 1, "bot": 2, "overlay": 3, "sub1": 4, "sub2": 5}


def get_domain(filename):
    """Extract domain from filename like web_000001_cloud_microsoft_top.jpg"""
    stem = Path(filename).stem
    parts = stem.split("_")
    section = parts[-1]
    domain = "_".join(parts[2:-1])
    return domain, section


def main():
    print("Scanning images...")
    domain_images = defaultdict(list)
    for img in SRC_IMAGES.iterdir():
        if not img.suffix == ".jpg":
            continue
        domain, section = get_domain(img.name)
        priority = SECTION_PRIORITY.get(section, 99)
        domain_images[domain].append((priority, img.name))

    # Pick best (lowest priority) image per domain
    selected = []
    for domain, images in domain_images.items():
        images.sort()  # Sort by priority
        selected.append(images[0][1])  # Best priority image

    selected.sort()
    print(f"Selected {len(selected)} images from {len(domain_images)} domains")

    # Create output directories
    DST_IMAGES.mkdir(parents=True, exist_ok=True)
    DST_LABELS.mkdir(parents=True, exist_ok=True)

    # Copy labels (small text files) and hardlink images (avoid copying 10GB)
    print("Creating diverse subset...")
    copied = 0
    skipped = 0
    for img_name in selected:
        label_name = Path(img_name).stem + ".txt"

        src_img = SRC_IMAGES / img_name
        src_lbl = SRC_LABELS / label_name
        dst_img = DST_IMAGES / img_name
        dst_lbl = DST_LABELS / label_name

        if not src_lbl.exists():
            skipped += 1
            continue

        # Hardlink image (no copy needed)
        if not dst_img.exists():
            try:
                os.link(str(src_img), str(dst_img))
            except OSError:
                shutil.copy2(src_img, dst_img)

        # Copy label
        if not dst_lbl.exists():
            shutil.copy2(src_lbl, dst_lbl)

        copied += 1
        if copied % 10000 == 0:
            print(f"  {copied} / {len(selected)}...")

    print(f"Done: {copied} image+label pairs, {skipped} skipped (no label)")

    # Write dataset.yaml
    yaml_content = f"""path: C:/Users/Alden Bernstein/desktop/COGS181-Final/data
train: crawled_1cls_diverse/images
val: raw_1cls/val/images
test: raw_1cls/test/images
nc: 1
names: [ui_element]
"""
    (DST_DIR / "dataset.yaml").write_text(yaml_content)
    print(f"Dataset yaml: {DST_DIR / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
