"""Render GROUND-TRUTH bounding boxes for the val samples.

Draws the label boxes (green) on the SAME sampled val images used by
render_bbox_occ.py, so GT can be compared side-by-side with the model preds.
Handles both YOLO bbox labels (cls xc yc w h) and polygon/seg labels
(cls x1 y1 x2 y2 ...) by taking the min/max envelope.

Output: runs_external_eval_occ/preview_gt/<dataset>/*.jpg
"""
import random
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
EXTERNAL_DIR = ROOT.parent / "data" / "data" / "external"
PIO = ROOT / "_pio_yolo"
OUT_DIR = ROOT / "runs_external_eval_occ" / "preview_gt"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
N_SAMPLE = 8
SEED = 42
BROILER_VAL_FRAC = 0.20
GT_COLOR = (0, 200, 0)      # BGR green


def log(m):
    print(m, flush=True)


def list_images(d):
    return [p for p in sorted(Path(d).iterdir()) if p.suffix.lower() in IMG_EXTS]


def evenly_spaced(items, k):
    if len(items) <= k:
        return items
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


def broiler_val_imgs():
    imgs = list_images(EXTERNAL_DIR / "broiler_instance_seg" / "train" / "images")
    rng = random.Random(SEED)
    shuffled = imgs[:]
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * BROILER_VAL_FRAC))
    return sorted(shuffled[:n_val], key=lambda p: p.name)


# (dataset name, image dir, label dir)
def sources():
    return {
        "pio_original_val": (
            list_images(PIO / "images" / "val"),
            PIO / "labels" / "val"),
        "broiler_instance_seg": (
            broiler_val_imgs(),
            EXTERNAL_DIR / "broiler_instance_seg" / "train" / "labels"),
        "chicken_detection_fum": (
            list_images(EXTERNAL_DIR / "chicken_detection_fum" / "valid" / "images"),
            EXTERNAL_DIR / "chicken_detection_fum" / "valid" / "labels"),
    }


def parse_label(txt_path, w, h):
    """Return list of pixel (x1,y1,x2,y2) from a YOLO bbox or polygon label."""
    boxes = []
    if not txt_path.exists():
        return boxes
    for line in txt_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        coords = list(map(float, parts[1:]))
        if len(coords) == 4:                      # bbox: xc yc bw bh
            xc, yc, bw, bh = coords
            x1 = (xc - bw / 2) * w
            y1 = (yc - bh / 2) * h
            x2 = (xc + bw / 2) * w
            y2 = (yc + bh / 2) * h
        else:                                     # polygon: x1 y1 x2 y2 ...
            xs = coords[0::2]
            ys = coords[1::2]
            x1, x2 = min(xs) * w, max(xs) * w
            y1, y2 = min(ys) * h, max(ys) * h
        boxes.append((int(x1), int(y1), int(x2), int(y2)))
    return boxes


def render(ds_name, imgs, label_dir):
    sample = evenly_spaced(imgs, N_SAMPLE)
    out = OUT_DIR / ds_name
    out.mkdir(parents=True, exist_ok=True)
    log(f"[{ds_name}] GT render {len(sample)} imgs -> {out}")
    for img in sample:
        im = cv2.imread(str(img))
        if im is None:
            log(f"  !! cannot read {img.name}")
            continue
        h, w = im.shape[:2]
        boxes = parse_label(label_dir / f"{img.stem}.txt", w, h)
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(im, (x1, y1), (x2, y2), GT_COLOR, 2)
        cv2.putText(im, f"GT: {len(boxes)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, GT_COLOR, 2, cv2.LINE_AA)
        dst = out / f"{img.stem}__gt{len(boxes)}.jpg"
        cv2.imwrite(str(dst), im)
        log(f"  {img.name}: {len(boxes)} GT boxes -> {dst.name}")
    log(f"[{ds_name}] done\n")


def main():
    for ds_name, (imgs, label_dir) in sources().items():
        render(ds_name, imgs, label_dir)
    log(f"ALL DONE -> {OUT_DIR}")


if __name__ == "__main__":
    main()
