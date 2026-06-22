from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from common import (
    COBB500_AS_HATCHED,
    CONFIG_DIR,
    DATA_DIR,
    REPORT_DIR,
    cobb_weight_for_age,
    ensure_dirs,
    image_size,
    iter_images,
    iter_labels,
    parse_filename_metadata,
    read_xlsx_first_sheet,
    read_yolo_label,
    write_csv,
    write_json,
)


def main() -> None:
    ensure_dirs()
    prefix_path = DATA_DIR / "FilePrefixCode.xlsx"
    prefix_rows = read_xlsx_first_sheet(prefix_path) if prefix_path.exists() else []
    write_csv(CONFIG_DIR / "prefix_mapping.csv", prefix_rows)

    cobb_rows = [{"age_days": d, "weight_g": w} for d, w in sorted(COBB500_AS_HATCHED.items())]
    write_csv(CONFIG_DIR / "cobb500_as_hatched.csv", cobb_rows)

    report = {
        "splits": {},
        "prefix_mapping_rows": len(prefix_rows),
        "cobb_reference": {
            "source": "2022 Cobb500 Broiler Performance & Nutrition Supplement, Metric As Hatched",
            "week_targets": {str(w): cobb_weight_for_age(w * 7) for w in range(1, 7)},
        },
    }

    for split in ["train", "val"]:
        images = iter_images(split)
        labels = iter_labels(split)
        image_stems = {p.stem for p in images}
        label_stems = {p.stem for p in labels}
        missing_labels = sorted(image_stems - label_stems)
        labels_without_images = sorted(label_stems - image_stems)

        week_counts = Counter()
        house_counts = Counter()
        image_sizes = Counter()
        bbox_count_per_image = {}
        invalid_bbox_rows = []
        unreadable_images = []

        for img in images:
            meta = parse_filename_metadata(img.name)
            week_counts[str(meta["week"])] += 1
            house_counts[str(meta["house"])] += 1
            try:
                image_sizes[str(image_size(img))] += 1
            except Exception as exc:
                unreadable_images.append({"image": str(img), "error": str(exc)})

            label = DATA_DIR / "labels" / split / f"{img.stem}.txt"
            try:
                bboxes = read_yolo_label(label)
            except Exception as exc:
                invalid_bbox_rows.append({"label": str(label), "error": str(exc)})
                bboxes = []
            bbox_count_per_image[img.name] = len(bboxes)
            for idx, (cls, x, y, w, h) in enumerate(bboxes):
                if cls != 0 or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    invalid_bbox_rows.append(
                        {"label": str(label), "row": idx + 1, "class": cls, "x": x, "y": y, "w": w, "h": h}
                    )

        counts = list(bbox_count_per_image.values())
        report["splits"][split] = {
            "images": len(images),
            "labels": len(labels),
            "missing_labels": missing_labels[:100],
            "labels_without_images": labels_without_images[:100],
            "missing_labels_count": len(missing_labels),
            "labels_without_images_count": len(labels_without_images),
            "week_image_counts": dict(sorted(week_counts.items())),
            "house_image_counts": dict(sorted(house_counts.items())),
            "image_sizes": dict(image_sizes.most_common(20)),
            "bbox_count": {
                "total": sum(counts),
                "min_per_image": min(counts) if counts else 0,
                "max_per_image": max(counts) if counts else 0,
                "mean_per_image": round(sum(counts) / len(counts), 2) if counts else 0,
            },
            "invalid_bbox_rows_count": len(invalid_bbox_rows),
            "invalid_bbox_rows_sample": invalid_bbox_rows[:50],
            "unreadable_images_count": len(unreadable_images),
            "unreadable_images_sample": unreadable_images[:20],
        }

    write_json(REPORT_DIR / "dataset_audit.json", report)
    print("Wrote reports/dataset_audit.json")
    print("Wrote configs/prefix_mapping.csv")
    print("Wrote configs/cobb500_as_hatched.csv")


if __name__ == "__main__":
    main()