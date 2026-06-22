"""
Convert common annotation formats to YOLO bounding-box .txt for the generalizability study.

Supported inputs:
  - COCO JSON           (--format coco)      bbox [x,y,w,h] absolute -> normalized YOLO
  - YOLO segmentation   (--format seg)       polygon points -> tight bbox
  - Pascal VOC XML      (--format voc)       xmin/ymin/xmax/ymax -> normalized YOLO

All outputs are written next to a new labels/ tree so the dataset becomes loadable by
validate_external_dataset.py and the main pipeline (which expects 5-col YOLO bbox).

Usage examples:
  python scripts/convert_to_yolo_bbox.py --format seg --root data/external/broiler_instance_seg
  python scripts/convert_to_yolo_bbox.py --format coco --coco-json data/external/x/_annotations.coco.json --images data/external/x/images --out data/external/x/labels
  python scripts/convert_to_yolo_bbox.py --format voc --root data/external/y

For seg: a YOLO segmentation label line is "<cls> x1 y1 x2 y2 ... xn yn" (normalized).
We collapse it to "<cls> xc yc w h" using the polygon's min/max.
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]


def seg_line_to_bbox(parts: List[str]) -> str | None:
    """parts = [cls, x1,y1, x2,y2, ...] normalized -> '<cls> xc yc w h'."""
    cls = int(float(parts[0]))
    coords = list(map(float, parts[1:]))
    if len(coords) < 6 or len(coords) % 2 != 0:
        return None
    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w = x_max - x_min
    h = y_max - y_min
    if w <= 0 or h <= 0:
        return None
    xc = x_min + w / 2
    yc = y_min + h / 2
    return f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def convert_seg(root: Path) -> dict:
    """In-place: rewrite any YOLO-seg label (>5 cols) under root into bbox labels.

    Backs up the original folder as labels_seg_backup if it is named 'labels'."""
    converted = 0
    skipped = 0
    files = 0
    for lbl in root.rglob("*.txt"):
        stem_lower = lbl.stem.lower()
        name_lower = lbl.name.lower()
        if stem_lower in {"classes", "data", "dataset", "readme", "notes"} or name_lower.startswith("readme"):
            continue
        lines_out = []
        changed = False
        try:
            raw = lbl.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        files += 1
        for line in raw:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) == 5:
                lines_out.append(line.strip())  # already bbox
            elif len(parts) > 5:
                bb = seg_line_to_bbox(parts)
                if bb:
                    lines_out.append(bb)
                    converted += 1
                    changed = True
                else:
                    skipped += 1
            else:
                skipped += 1
        if changed:
            lbl.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    return {"files_scanned": files, "polygons_converted": converted, "lines_skipped": skipped}


def convert_coco(coco_json: Path, images_dir: Path, out_dir: Path) -> dict:
    data = json.loads(coco_json.read_text(encoding="utf-8"))
    images = {im["id"]: im for im in data.get("images", [])}
    out_dir.mkdir(parents=True, exist_ok=True)

    # remap category ids to contiguous 0-based class ids
    cats = sorted({a["category_id"] for a in data.get("annotations", [])})
    cat_map = {c: i for i, c in enumerate(cats)}

    per_image_lines: dict = {}
    n = 0
    for ann in data.get("annotations", []):
        im = images.get(ann["image_id"])
        if not im:
            continue
        W, H = im["width"], im["height"]
        x, y, w, h = ann["bbox"]  # absolute top-left
        if w <= 0 or h <= 0:
            continue
        xc = (x + w / 2) / W
        yc = (y + h / 2) / H
        cls = cat_map[ann["category_id"]]
        per_image_lines.setdefault(im["file_name"], []).append(
            f"{cls} {xc:.6f} {yc:.6f} {w / W:.6f} {h / H:.6f}")
        n += 1

    for fname, lines in per_image_lines.items():
        stem = Path(fname).stem
        (out_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {"annotations": n, "images_with_labels": len(per_image_lines),
            "classes": len(cats), "category_map": cat_map}


def convert_voc(root: Path) -> dict:
    out = 0
    boxes = 0
    classes: dict = {}
    for xml in root.rglob("*.xml"):
        try:
            tree = ET.parse(xml)
        except Exception:
            continue
        r = tree.getroot()
        size = r.find("size")
        if size is None:
            continue
        W = float(size.findtext("width", "0"))
        H = float(size.findtext("height", "0"))
        if W <= 0 or H <= 0:
            continue
        lines = []
        for obj in r.findall("object"):
            name = obj.findtext("name", "object")
            cls = classes.setdefault(name, len(classes))
            b = obj.find("bndbox")
            if b is None:
                continue
            xmin = float(b.findtext("xmin", "0"))
            ymin = float(b.findtext("ymin", "0"))
            xmax = float(b.findtext("xmax", "0"))
            ymax = float(b.findtext("ymax", "0"))
            w = xmax - xmin
            h = ymax - ymin
            if w <= 0 or h <= 0:
                continue
            xc = (xmin + w / 2) / W
            yc = (ymin + h / 2) / H
            lines.append(f"{cls} {xc:.6f} {yc:.6f} {w / W:.6f} {h / H:.6f}")
            boxes += 1
        if lines:
            (xml.with_suffix(".txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")
            out += 1
    return {"xml_converted": out, "boxes": boxes, "class_map": classes}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", required=True, choices=["coco", "seg", "voc"])
    ap.add_argument("--root", help="dataset root (for seg/voc in-place conversion)")
    ap.add_argument("--coco-json", help="path to COCO annotations json")
    ap.add_argument("--images", help="images dir (coco)")
    ap.add_argument("--out", help="output labels dir (coco)")
    args = ap.parse_args()

    if args.format == "seg":
        if not args.root:
            raise SystemExit("--root required for seg")
        res = convert_seg(Path(args.root))
    elif args.format == "voc":
        if not args.root:
            raise SystemExit("--root required for voc")
        res = convert_voc(Path(args.root))
    else:  # coco
        if not (args.coco_json and args.images and args.out):
            raise SystemExit("--coco-json, --images, --out required for coco")
        res = convert_coco(Path(args.coco_json), Path(args.images), Path(args.out))

    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
