from __future__ import annotations

import math
from pathlib import Path

from common import (
    DATA_DIR,
    FEATURE_DIR,
    cobb_weight_for_age,
    ensure_dirs,
    image_size,
    iter_images,
    parse_filename_metadata,
    read_yolo_label,
    write_csv,
)


FIELDS = [
    "split",
    "image",
    "label",
    "bbox_id",
    "house_code",
    "house",
    "week",
    "age_days",
    "cobb_weight_g",
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


def main() -> None:
    ensure_dirs()
    rows = []
    skipped = []

    for split in ["train", "val"]:
        for img in iter_images(split):
            label = DATA_DIR / "labels" / split / f"{img.stem}.txt"
            if not label.exists():
                skipped.append({"image": str(img), "reason": "missing_label"})
                continue
            try:
                img_w, img_h = image_size(img)
                boxes = read_yolo_label(label)
            except Exception as exc:
                skipped.append({"image": str(img), "reason": str(exc)})
                continue

            meta = parse_filename_metadata(img.name)
            age_days = meta["age_days"]
            cobb = cobb_weight_for_age(age_days) if age_days is not None else None
            if meta["week"] is None or cobb is None:
                skipped.append({"image": str(img), "reason": "filename_not_in_FilePrefixCode_pattern_or_no_Cobb500_week"})
                continue

            for bbox_id, (cls, x, y, w, h) in enumerate(boxes):
                if cls != 0 or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    skipped.append({"image": str(img), "label": str(label), "bbox_id": bbox_id, "reason": "invalid_yolo_bbox"})
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
                rows.append(
                    {
                        "split": split,
                        "image": img.name,
                        "label": label.name,
                        "bbox_id": bbox_id,
                        "house_code": meta["house_code"],
                        "house": meta["house"],
                        "week": meta["week"],
                        "age_days": age_days,
                        "cobb_weight_g": cobb,
                        "class_id": cls,
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

    write_csv(FEATURE_DIR / "bbox_features.csv", rows, FIELDS)
    write_csv(FEATURE_DIR / "bbox_feature_skips.csv", skipped)
    print(f"Wrote features/bbox_features.csv ({len(rows)} rows)")
    if skipped:
        print(f"Wrote features/bbox_feature_skips.csv ({len(skipped)} rows)")


if __name__ == "__main__":
    main()