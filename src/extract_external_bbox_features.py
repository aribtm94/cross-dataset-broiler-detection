from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT, FEATURE_DIR, image_size, read_yolo_label, write_csv, write_json  # noqa: E402


FIELDS = [
    "dataset_id",
    "mode",
    "source_split",
    "image",
    "image_relpath",
    "label_relpath",
    "bbox_id",
    "class_id",
    "x_center_norm",
    "y_center_norm",
    "w_norm",
    "h_norm",
    "image_width",
    "image_height",
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x_px",
    "center_y_px",
    "bottom_y_norm",
    "radius_from_center_px",
    "radius_norm",
    "width_px",
    "height_px",
    "minor_axis",
    "major_axis",
    "ellipse_area",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_config(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def split_name(image_dir: str) -> str:
    parts = Path(image_dir).parts
    for p in parts:
        if p.lower() in {"train", "valid", "val", "test"}:
            return "val" if p.lower() == "valid" else p.lower()
    return Path(image_dir).name or "all"


def iter_pairs(dataset: Dict[str, Any]) -> Iterable[Tuple[str, Path, Path]]:
    root = ROOT / dataset["root"] if not Path(dataset["root"]).is_absolute() else Path(dataset["root"])
    image_dirs = dataset.get("image_dirs") or []
    label_dirs = dataset.get("label_dirs") or []
    if len(image_dirs) != len(label_dirs):
        raise ValueError(f"{dataset['dataset_id']}: image_dirs and label_dirs length mismatch")

    for image_dir, label_dir in zip(image_dirs, label_dirs):
        img_base = root / image_dir
        lbl_base = root / label_dir
        source_split = split_name(image_dir)
        if not img_base.exists():
            continue
        for img in sorted(p for p in img_base.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS):
            label = lbl_base / f"{img.stem}.txt"
            yield source_split, img, label


def should_include_class(cls: int, dataset: Dict[str, Any]) -> bool:
    policy = dataset.get("class_policy", "include_classes")
    if policy == "collapse_to_chicken":
        return True
    include = dataset.get("include_classes")
    if include is None:
        return True
    return cls in set(int(x) for x in include)


def extract_dataset(dataset: Dict[str, Any]) -> Dict[str, Any]:
    dataset_id = dataset["dataset_id"]
    mode = dataset.get("mode", "relative")
    root = ROOT / dataset["root"] if not Path(dataset["root"]).is_absolute() else Path(dataset["root"])
    out_dir = FEATURE_DIR / "external" / dataset_id
    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    images_seen = 0

    for source_split, img, label in iter_pairs(dataset):
        images_seen += 1
        if not label.exists():
            skipped.append({"dataset_id": dataset_id, "image": str(img), "reason": "missing_label"})
            continue
        try:
            img_w, img_h = image_size(img)
            boxes = read_yolo_label(label)
        except Exception as exc:
            skipped.append({"dataset_id": dataset_id, "image": str(img), "label": str(label), "reason": str(exc)})
            continue

        for bbox_id, (cls, x, y, w, h) in enumerate(boxes):
            if not should_include_class(cls, dataset):
                skipped.append({"dataset_id": dataset_id, "image": str(img), "label": str(label), "bbox_id": bbox_id, "reason": "class_excluded"})
                continue
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                skipped.append({"dataset_id": dataset_id, "image": str(img), "label": str(label), "bbox_id": bbox_id, "reason": "invalid_yolo_bbox"})
                continue

            width_px = w * img_w
            height_px = h * img_h
            minor = min(width_px, height_px)
            major = max(width_px, height_px)
            cx = x * img_w
            cy = y * img_h
            x1 = max(0.0, cx - width_px / 2)
            y1 = max(0.0, cy - height_px / 2)
            x2 = min(float(img_w), cx + width_px / 2)
            y2 = min(float(img_h), cy + height_px / 2)
            radius = math.sqrt((cx - img_w / 2) ** 2 + (cy - img_h / 2) ** 2)
            max_radius = math.sqrt((img_w / 2) ** 2 + (img_h / 2) ** 2)
            class_id = 0 if dataset.get("class_policy") == "collapse_to_chicken" else cls

            rows.append(
                {
                    "dataset_id": dataset_id,
                    "mode": mode,
                    "source_split": source_split,
                    "image": img.name,
                    "image_relpath": str(img.relative_to(root)),
                    "label_relpath": str(label.relative_to(root)),
                    "bbox_id": bbox_id,
                    "class_id": class_id,
                    "x_center_norm": round(x, 8),
                    "y_center_norm": round(y, 8),
                    "w_norm": round(w, 8),
                    "h_norm": round(h, 8),
                    "image_width": img_w,
                    "image_height": img_h,
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                    "center_x_px": round(cx, 2),
                    "center_y_px": round(cy, 2),
                    "bottom_y_norm": round(y2 / img_h, 8),
                    "radius_from_center_px": round(radius, 4),
                    "radius_norm": round(radius / max_radius, 8),
                    "width_px": round(width_px, 4),
                    "height_px": round(height_px, 4),
                    "minor_axis": round(minor, 4),
                    "major_axis": round(major, 4),
                    "ellipse_area": round(math.pi * minor * major, 4),
                }
            )

    write_csv(out_dir / "bbox_features.csv", rows, FIELDS)
    write_csv(out_dir / "bbox_feature_skips.csv", skipped)
    summary = {
        "dataset_id": dataset_id,
        "display_name": dataset.get("display_name", dataset_id),
        "mode": mode,
        "images_seen": images_seen,
        "valid_bboxes": len(rows),
        "skipped": len(skipped),
        "output": str(out_dir / "bbox_features.csv"),
    }
    write_json(out_dir / "bbox_feature_summary.json", summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "datasets" / "external_datasets.json"))
    ap.add_argument("--dataset", default="", help="dataset_id to process; default all")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    summaries = []
    for dataset in cfg.get("datasets", []):
        if args.dataset and dataset["dataset_id"] != args.dataset:
            continue
        summary = extract_dataset(dataset)
        summaries.append(summary)
        print(f"{summary['dataset_id']}: images={summary['images_seen']} bboxes={summary['valid_bboxes']} skipped={summary['skipped']}")
    if not summaries:
        raise SystemExit(f"No datasets matched {args.dataset!r}")


if __name__ == "__main__":
    main()
