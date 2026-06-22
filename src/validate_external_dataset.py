"""
Validate an external broiler/chicken detection dataset for the generalizability study.

Generic YOLO-dataset auditor. Works on any folder that contains images + YOLO .txt
labels, regardless of internal layout. It auto-discovers image/label pairs, computes
bbox statistics, detects annotation format issues, and writes a JSON + CSV report.

Usage:
    python scripts/validate_external_dataset.py --name broiler_roboflow --root data/external/broiler_roboflow
    python scripts/validate_external_dataset.py --name pio --root data --pio   # validate the baseline PIO too

Output:
    reports/external/<name>_audit.json
    reports/external/<name>_image_stats.csv

Design notes:
- Reuses image_size / read_yolo_label from common.py (no extra dependencies).
- Pairs an image to its label by matching the file stem anywhere under the root.
- A label whose matching image is missing (or vice versa) is reported, not fatal.
- Handles datasets with multiple classes (Roboflow often has >1 class); records the
  class distribution so we can decide later which class == "chicken body".
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
from common import image_size, read_yolo_label, write_csv, write_json, mean, median, stdev, percentile  # noqa: E402


ROOT = common.ROOT
IMAGE_EXTS = common.IMAGE_EXTS


def find_files(root: Path) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    """Walk the dataset root, return {stem: image_path} and {stem: label_path}.

    A 'classes.txt' / 'data.yaml' style file is ignored as a label (it has no image)."""
    images: Dict[str, Path] = {}
    labels: Dict[str, Path] = {}
    skip_label_names = {"classes", "data", "dataset", "readme", "notes"}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        suffix = p.suffix.lower()
        if suffix in IMAGE_EXTS:
            # later wins only if duplicate stems exist; record first to be deterministic
            images.setdefault(p.stem, p)
        elif suffix == ".txt":
            if p.stem.lower() in skip_label_names:
                continue
            labels.setdefault(p.stem, p)
    return images, labels


def validate(name: str, root: Path) -> dict:
    root = root.resolve()
    if not root.exists():
        raise SystemExit(f"Dataset root not found: {root}")

    images, labels = find_files(root)
    image_stems = set(images)
    label_stems = set(labels)

    paired = sorted(image_stems & label_stems)
    images_without_label = sorted(image_stems - label_stems)
    labels_without_image = sorted(label_stems - image_stems)

    class_counter: Counter = Counter()
    bbox_areas_norm: List[float] = []
    bbox_w_norm: List[float] = []
    bbox_h_norm: List[float] = []
    bbox_per_image: List[int] = []
    resolutions: Counter = Counter()

    invalid_bbox = 0
    unreadable_images = 0
    unreadable_labels = 0
    per_image_rows: List[dict] = []

    for stem in paired:
        img_path = images[stem]
        lbl_path = labels[stem]

        # image size
        try:
            w_px, h_px = image_size(img_path)
            resolutions[f"{w_px}x{h_px}"] += 1
        except Exception:
            unreadable_images += 1
            w_px = h_px = 0

        # labels
        try:
            rows = read_yolo_label(lbl_path)
        except Exception:
            unreadable_labels += 1
            rows = []

        n_valid = 0
        for cls, x, y, bw, bh in rows:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0) or bw <= 0 or bh <= 0:
                invalid_bbox += 1
                continue
            class_counter[cls] += 1
            bbox_w_norm.append(bw)
            bbox_h_norm.append(bh)
            bbox_areas_norm.append(bw * bh)
            n_valid += 1

        bbox_per_image.append(n_valid)
        per_image_rows.append({
            "stem": stem,
            "image": str(img_path.relative_to(root)),
            "width_px": w_px,
            "height_px": h_px,
            "bbox_count": n_valid,
        })

    def stats(vals: List[float]) -> dict:
        if not vals:
            return {"n": 0}
        return {
            "n": len(vals),
            "mean": round(mean(vals) or 0, 6),
            "median": round(median(vals) or 0, 6),
            "stdev": round(stdev(vals), 6),
            "min": round(min(vals), 6),
            "max": round(max(vals), 6),
            "p05": round(percentile(vals, 0.05) or 0, 6),
            "p95": round(percentile(vals, 0.95) or 0, 6),
        }

    total_bbox = sum(class_counter.values())
    report = {
        "dataset_name": name,
        "root": str(root),
        "counts": {
            "images_total": len(images),
            "labels_total": len(labels),
            "paired": len(paired),
            "images_without_label": len(images_without_label),
            "labels_without_image": len(labels_without_image),
            "total_valid_bbox": total_bbox,
            "invalid_bbox_skipped": invalid_bbox,
            "unreadable_images": unreadable_images,
            "unreadable_labels": unreadable_labels,
        },
        "class_distribution": {str(k): v for k, v in sorted(class_counter.items())},
        "num_classes_observed": len(class_counter),
        "bbox_per_image": stats([float(x) for x in bbox_per_image]),
        "bbox_width_norm": stats(bbox_w_norm),
        "bbox_height_norm": stats(bbox_h_norm),
        "bbox_area_norm": stats(bbox_areas_norm),
        "top_resolutions": dict(resolutions.most_common(10)),
        "num_distinct_resolutions": len(resolutions),
        "examples_images_without_label": images_without_label[:10],
        "examples_labels_without_image": labels_without_image[:10],
        # heuristic readiness flags for the pipeline
        "pipeline_readiness": {
            "is_yolo_format": len(paired) > 0,
            "single_class": len(class_counter) <= 1,
            "has_high_density": (median([float(x) for x in bbox_per_image]) or 0) >= 20,
            "uniform_resolution": len(resolutions) <= 3,
        },
    }

    out_dir = ROOT / "reports" / "external"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / f"{name}_audit.json", report)
    write_csv(out_dir / f"{name}_image_stats.csv", per_image_rows,
              fieldnames=["stem", "image", "width_px", "height_px", "bbox_count"])

    return report


def print_summary(report: dict) -> None:
    c = report["counts"]
    print(f"\n=== Dataset: {report['dataset_name']} ===")
    print(f"root: {report['root']}")
    print(f"images={c['images_total']} labels={c['labels_total']} paired={c['paired']}")
    print(f"images_without_label={c['images_without_label']} labels_without_image={c['labels_without_image']}")
    print(f"total_valid_bbox={c['total_valid_bbox']} invalid_skipped={c['invalid_bbox_skipped']}")
    print(f"classes observed: {report['class_distribution']} (n={report['num_classes_observed']})")
    bpi = report["bbox_per_image"]
    if bpi.get("n"):
        print(f"bbox/image: mean={bpi['mean']} median={bpi['median']} max={bpi['max']}")
    print(f"distinct resolutions: {report['num_distinct_resolutions']}, top: {list(report['top_resolutions'].items())[:3]}")
    r = report["pipeline_readiness"]
    print(f"readiness: yolo={r['is_yolo_format']} single_class={r['single_class']} "
          f"high_density={r['has_high_density']} uniform_res={r['uniform_resolution']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="short dataset id, e.g. broiler_roboflow")
    ap.add_argument("--root", required=True, help="path to dataset root folder")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_absolute():
        root = ROOT / root
    report = validate(args.name, root)
    print_summary(report)
    print(f"\nWrote reports/external/{args.name}_audit.json")


if __name__ == "__main__":
    main()
