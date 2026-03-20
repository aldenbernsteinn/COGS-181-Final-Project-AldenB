"""Generate YOLO training data by crawling real websites with Playwright.

Visits real websites from the Tranco Top 1M list, takes screenshots at 1920x1080,
and extracts all interactive UI elements from the DOM as YOLO annotations.

Usage:
    python generate_training_data.py                        # Crawl 70K sites
    python generate_training_data.py --num-pages 1000       # Crawl 1000 sites
    python generate_training_data.py --workers 8            # 8 parallel browsers
    python generate_training_data.py --resume               # Resume from last run
"""

import argparse
import asyncio
import csv
import io
import json
import os
import random
import time
import zipfile
from pathlib import Path

import requests
from playwright.async_api import async_playwright

# ─── Constants ────────────────────────────────────────────────────────────────

WIDTH, HEIGHT = 1920, 1080
NUM_PAGES_DEFAULT = 70_000
NUM_WORKERS_DEFAULT = 8
PAGE_TIMEOUT_MS = 15_000  # 15s max per page
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
OUTPUT_DIR = Path("data/crawled")
TRANCO_URL = "https://tranco-list.eu/top-1m.csv.zip"
TRANCO_CACHE = Path("data/tranco_top1m.csv")
MIN_ELEMENTS = 3  # Skip pages with fewer elements
PROGRESS_FILE = OUTPUT_DIR / "progress.json"

# YOLO classes (matching data/raw/dataset.yaml)
# 0:link 1:button 2:input 3:select 4:textarea 5:label
# 6:checkbox 7:radio 8:dropdown 9:slider 10:toggle 11:menu_item
# 12:clickable 13:icon 14:image 15:text

# ─── JavaScript element extraction ───────────────────────────────────────────

EXTRACT_ELEMENTS_JS = """
() => {
    const W = window.innerWidth;
    const H = window.innerHeight;
    const results = [];
    const seen = new Set();

    function isVisible(el) {
        const style = window.getComputedStyle(el);
        if (style.display === 'none' || style.visibility === 'hidden') return false;
        if (parseFloat(style.opacity) < 0.1) return false;
        return true;
    }

    function addElement(el, classId) {
        if (!isVisible(el)) return;
        const rect = el.getBoundingClientRect();
        if (rect.width < 4 || rect.height < 4) return;
        if (rect.right < 0 || rect.bottom < 0 || rect.left > W || rect.top > H) return;

        const x1 = Math.max(0, rect.left);
        const y1 = Math.max(0, rect.top);
        const x2 = Math.min(W, rect.right);
        const y2 = Math.min(H, rect.bottom);
        const w = x2 - x1;
        const h = y2 - y1;
        if (w < 4 || h < 4) return;

        const cx = (x1 + w / 2) / W;
        const cy = (y1 + h / 2) / H;
        const nw = w / W;
        const nh = h / H;

        // Deduplicate by position (within 2px)
        const key = `${classId}_${Math.round(cx*500)}_${Math.round(cy*500)}`;
        if (seen.has(key)) return;
        seen.add(key);

        results.push({ c: classId, cx: cx, cy: cy, w: nw, h: nh });
    }

    // Class 0: Links
    document.querySelectorAll('a[href]').forEach(el => {
        // If inside nav/header, classify as menu_item (11) instead
        if (el.closest('nav, header, [role="navigation"], [role="menubar"]')) {
            addElement(el, 11);
        } else {
            addElement(el, 0);
        }
    });

    // Class 1: Buttons
    document.querySelectorAll('button, input[type="submit"], input[type="button"], [role="button"]').forEach(el => {
        if (el.tagName === 'A') return;  // already counted as link
        addElement(el, 1);
    });

    // Class 2: Text inputs
    document.querySelectorAll('input[type="text"], input[type="email"], input[type="password"], input[type="number"], input[type="search"], input[type="tel"], input[type="url"], input:not([type])').forEach(el => {
        addElement(el, 2);
    });

    // Class 3: Select dropdowns
    document.querySelectorAll('select').forEach(el => addElement(el, 3));

    // Class 4: Textareas
    document.querySelectorAll('textarea').forEach(el => addElement(el, 4));

    // Class 5: Labels
    document.querySelectorAll('label').forEach(el => addElement(el, 5));

    // Class 6: Checkboxes
    document.querySelectorAll('input[type="checkbox"]').forEach(el => addElement(el, 6));

    // Class 7: Radio buttons
    document.querySelectorAll('input[type="radio"]').forEach(el => addElement(el, 7));

    // Class 8: Custom dropdowns (common patterns)
    document.querySelectorAll('[class*="dropdown"], [class*="Dropdown"], [data-dropdown], [role="listbox"], [role="combobox"], [class*="select-menu"], [class*="selectMenu"]').forEach(el => {
        if (el.tagName !== 'SELECT') addElement(el, 8);
    });

    // Class 9: Sliders/Range inputs
    document.querySelectorAll('input[type="range"], [role="slider"], [class*="slider" i], [class*="range" i]').forEach(el => {
        if (el.tagName === 'INPUT' && el.type === 'range') addElement(el, 9);
        else if (el.tagName !== 'INPUT') addElement(el, 9);
    });

    // Class 10: Toggle switches
    document.querySelectorAll('[role="switch"], [class*="toggle" i], [class*="switch" i]').forEach(el => {
        // Avoid matching unrelated "toggle" classes (like toggle-menu)
        const rect = el.getBoundingClientRect();
        if (rect.width > 10 && rect.width < 120 && rect.height > 10 && rect.height < 60) {
            addElement(el, 10);
        }
    });

    // Class 11: Menu items (already handled above for nav links)
    document.querySelectorAll('[role="menuitem"], [role="tab"], li > a').forEach(el => {
        addElement(el, 11);
    });

    // Class 12: Generic clickable elements
    document.querySelectorAll('[onclick], [tabindex="0"], [role="link"], [class*="clickable" i], [class*="card" i][onclick], [data-click]').forEach(el => {
        if (!['A', 'BUTTON', 'INPUT', 'SELECT', 'TEXTAREA', 'LABEL'].includes(el.tagName)) {
            addElement(el, 12);
        }
    });

    // Class 13: Icons
    document.querySelectorAll('svg, i[class], span[class*="icon" i], img[class*="icon" i], [class*="icon" i]:not(div):not(span):not(a):not(button), [role="img"][aria-label]').forEach(el => {
        const rect = el.getBoundingClientRect();
        // Icons are typically small
        if (rect.width >= 8 && rect.width <= 80 && rect.height >= 8 && rect.height <= 80) {
            addElement(el, 13);
        }
    });

    // Also catch SVGs inside buttons/links that are icon-sized
    document.querySelectorAll('button svg, a svg').forEach(el => {
        const rect = el.getBoundingClientRect();
        if (rect.width >= 8 && rect.width <= 60 && rect.height >= 8 && rect.height <= 60) {
            addElement(el, 13);
        }
    });

    // Class 14: Images
    document.querySelectorAll('img, picture, [role="img"], video, canvas').forEach(el => {
        const rect = el.getBoundingClientRect();
        // Filter out tiny tracking pixels
        if (rect.width >= 20 && rect.height >= 20) {
            addElement(el, 14);
        }
    });

    // Class 15: Text blocks
    document.querySelectorAll('h1, h2, h3, h4, h5, h6').forEach(el => {
        if (!el.closest('a') && !el.closest('button')) addElement(el, 15);
    });
    document.querySelectorAll('p').forEach(el => {
        if (!el.closest('a') && !el.closest('button') && el.textContent.trim().length > 5) {
            addElement(el, 15);
        }
    });

    return results;
}
"""

# ─── Tranco list download ────────────────────────────────────────────────────


def download_tranco_list():
    """Download and cache the Tranco Top 1M website list."""
    if TRANCO_CACHE.exists():
        print(f"Using cached Tranco list: {TRANCO_CACHE}")
    else:
        print(f"Downloading Tranco Top 1M list...")
        TRANCO_CACHE.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(TRANCO_URL, timeout=60)
        resp.raise_for_status()

        # Tranco delivers a ZIP containing a CSV
        if resp.content[:2] == b'PK':
            z = zipfile.ZipFile(io.BytesIO(resp.content))
            csv_name = z.namelist()[0]
            z.extract(csv_name, TRANCO_CACHE.parent)
            extracted = TRANCO_CACHE.parent / csv_name
            extracted.rename(TRANCO_CACHE)
        else:
            TRANCO_CACHE.write_bytes(resp.content)
        print(f"Saved: {TRANCO_CACHE}")

    # Parse domains
    domains = []
    with open(TRANCO_CACHE, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                domains.append(row[1].strip())
            elif len(row) == 1:
                domains.append(row[0].strip())
    print(f"Loaded {len(domains)} domains")
    return domains


# ─── Rendering pipeline ──────────────────────────────────────────────────────


async def crawl_page(context, domain, page_idx, split):
    """Visit a real website and extract YOLO annotations."""
    url = f"https://{domain}"
    page = await context.new_page()

    try:
        # Navigate with timeout
        await page.goto(url, timeout=PAGE_TIMEOUT_MS, wait_until="domcontentloaded")

        # Wait a bit for dynamic content
        try:
            await page.wait_for_load_state("networkidle", timeout=5000)
        except:
            pass  # Some sites never reach networkidle, that's ok

        # Dismiss common popups/cookie banners
        for selector in [
            'button:has-text("Accept")', 'button:has-text("I agree")',
            'button:has-text("Got it")', 'button:has-text("OK")',
            'button:has-text("Close")', '[class*="cookie"] button',
            '[id*="cookie"] button', '[class*="consent"] button',
        ]:
            try:
                btn = page.locator(selector).first
                if await btn.is_visible(timeout=500):
                    await btn.click(timeout=1000)
                    await page.wait_for_timeout(300)
                    break
            except:
                continue

        # Extract element annotations
        elements = await page.evaluate(EXTRACT_ELEMENTS_JS)

        if len(elements) < MIN_ELEMENTS:
            await page.close()
            return None, "too_few_elements"

        # Save screenshot
        img_dir = OUTPUT_DIR / split / "images"
        label_dir = OUTPUT_DIR / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        # Use domain-based filename (sanitized)
        safe_name = domain.replace(".", "_").replace("/", "_")[:60]
        filename = f"web_{page_idx:06d}_{safe_name}"
        img_path = img_dir / f"{filename}.jpg"
        label_path = label_dir / f"{filename}.txt"

        await page.screenshot(path=str(img_path), type="jpeg", quality=85)

        # Save YOLO annotations
        lines = []
        for el in elements:
            lines.append(f"{el['c']} {el['cx']:.6f} {el['cy']:.6f} {el['w']:.6f} {el['h']:.6f}")

        with open(label_path, "w") as f:
            f.write("\n".join(lines))

        await page.close()
        return len(elements), None

    except Exception as e:
        try:
            await page.close()
        except:
            pass
        err_type = type(e).__name__
        return None, err_type


async def worker(worker_id, browser, tasks, progress, lock):
    """Worker that crawls assigned websites."""
    context = await browser.new_context(
        viewport={"width": WIDTH, "height": HEIGHT},
        device_scale_factor=1,
        locale="en-US",
        timezone_id="America/New_York",
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    )
    # Block heavy resources to speed up crawling
    await context.route("**/*.{mp4,webm,ogg,mp3,wav,flac}", lambda route: route.abort())
    await context.route("**/*.{woff2,woff,ttf,eot}", lambda route: route.abort())

    for domain, page_idx, split in tasks:
        n_elements, error = await crawl_page(context, domain, page_idx, split)

        async with lock:
            progress["done"] += 1
            if n_elements:
                progress["success"] += 1
                progress["elements"] += n_elements
            else:
                progress["errors"][error] = progress["errors"].get(error, 0) + 1

            if progress["done"] % 50 == 0:
                elapsed = time.time() - progress["start"]
                rate = progress["done"] / elapsed if elapsed > 0 else 0
                remaining = (progress["total"] - progress["done"]) / rate if rate > 0 else 0
                success_pct = progress["success"] / progress["done"] * 100 if progress["done"] > 0 else 0
                print(f"  [{progress['done']}/{progress['total']}] "
                      f"{rate:.1f} pages/sec | {success_pct:.0f}% success | "
                      f"~{remaining/60:.0f} min left | "
                      f"{progress['elements']} elements | "
                      f"worker {worker_id}")

            # Save progress periodically
            if progress["done"] % 500 == 0:
                save_progress(progress)

    await context.close()


def save_progress(progress):
    """Save crawl progress for resume capability."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump({
            "done": progress["done"],
            "success": progress["success"],
            "elements": progress["elements"],
            "errors": progress["errors"],
            "last_index": progress.get("last_index", 0),
        }, f)


def load_progress():
    """Load previous progress for resume."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return None


async def generate_all(num_pages, num_workers, resume=False):
    """Crawl real websites and generate YOLO training data."""
    print(f"Crawling {num_pages} real websites with {num_workers} workers...")
    print(f"Output: {OUTPUT_DIR}/")

    # Download/load domain list
    domains = download_tranco_list()

    # Determine start index
    start_idx = 0
    if resume:
        prev = load_progress()
        if prev:
            start_idx = prev.get("last_index", 0)
            print(f"Resuming from index {start_idx} ({prev['success']} already crawled)")

    # We need more domains than num_pages because some will fail
    # Use 1.5x to account for ~33% failure rate
    end_idx = min(start_idx + int(num_pages * 1.5), len(domains))
    selected_domains = domains[start_idx:end_idx]

    # Shuffle and assign splits (deterministic per domain)
    random.seed(42)
    tasks = []
    for i, domain in enumerate(selected_domains):
        page_idx = start_idx + i
        r = random.random()
        if r < SPLIT_RATIOS["train"]:
            split = "train"
        elif r < SPLIT_RATIOS["train"] + SPLIT_RATIOS["val"]:
            split = "val"
        else:
            split = "test"
        tasks.append((domain, page_idx, split))

    print(f"  Domains to visit: {len(tasks)} (expecting ~{num_pages} successful)")

    # Create output directories
    for split in SPLIT_RATIOS:
        (OUTPUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    progress = {
        "done": 0, "total": len(tasks), "success": 0,
        "elements": 0, "errors": {}, "start": time.time(),
        "last_index": end_idx,
    }
    lock = asyncio.Lock()

    # Distribute tasks across workers
    worker_tasks = [[] for _ in range(num_workers)]
    for i, task in enumerate(tasks):
        worker_tasks[i % num_workers].append(task)

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--disable-gpu", "--no-sandbox", "--disable-dev-shm-usage"],
        )

        workers = [
            worker(i, browser, worker_tasks[i], progress, lock)
            for i in range(num_workers)
        ]
        await asyncio.gather(*workers)

        await browser.close()

    elapsed = time.time() - progress["start"]
    print(f"\n{'='*60}")
    print(f"CRAWL COMPLETE")
    print(f"{'='*60}")
    print(f"Visited: {progress['done']} sites in {elapsed/3600:.1f} hours")
    print(f"Successful: {progress['success']} ({progress['success']/progress['done']*100:.0f}%)")
    print(f"Elements annotated: {progress['elements']}")
    print(f"Avg elements/page: {progress['elements']/max(1,progress['success']):.0f}")
    print(f"\nErrors:")
    for err, count in sorted(progress["errors"].items(), key=lambda x: -x[1]):
        print(f"  {err}: {count}")

    # Save final progress
    save_progress(progress)

    # Count per split
    for split in SPLIT_RATIOS:
        img_dir = OUTPUT_DIR / split / "images"
        if img_dir.exists():
            n = len(list(img_dir.glob("*.jpg")))
            print(f"\n{split}: {n} images")

    # Write dataset.yaml
    write_dataset_yaml()


def write_dataset_yaml():
    """Write dataset.yaml for the crawled dataset."""
    abs_path = OUTPUT_DIR.resolve().as_posix()
    yaml_content = f"""# Crawled real website dataset
path: {abs_path}
train: train/images
val: val/images
test: test/images

names:
  0: link
  1: button
  2: input
  3: select
  4: textarea
  5: label
  6: checkbox
  7: radio
  8: dropdown
  9: slider
  10: toggle
  11: menu_item
  12: clickable
  13: icon
  14: image
  15: text

nc: 16
"""
    with open(OUTPUT_DIR / "dataset.yaml", "w") as f:
        f.write(yaml_content)
    print(f"\nWrote {OUTPUT_DIR / 'dataset.yaml'}")


def main():
    parser = argparse.ArgumentParser(description="Crawl real websites for YOLO training data")
    parser.add_argument("--num-pages", type=int, default=NUM_PAGES_DEFAULT,
                        help=f"Target number of successful pages (default: {NUM_PAGES_DEFAULT})")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS_DEFAULT,
                        help=f"Number of parallel workers (default: {NUM_WORKERS_DEFAULT})")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last crawl")
    args = parser.parse_args()

    asyncio.run(generate_all(args.num_pages, args.workers, args.resume))


if __name__ == "__main__":
    main()
