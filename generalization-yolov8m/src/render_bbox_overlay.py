"""Overlay GROUND-TRUTH (green) + MODEL PREDICTIONS (blue) on one frame.

Same sampled val images as render_bbox_occ.py / render_bbox_gt.py, so each
overlay directly shows where the occlusion-aug model agrees with / diverges
from the labels. Predictions use max_det=1000 so dense scenes are not capped.

Output: runs_external_eval_occ/preview_overlay/<dataset>/*.jpg
"""
import random
from pathlib import Path

import cv2
import torch
torch.set_num_threads(2)

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
MODEL_PATH = (ROOT / "cmp_yolov8m_occ_rect" / "cmp_yolov8m_occ_rect"
              / "weights" / "best.pt")
EXTERNAL_DIR = ROOT.parent / "data" / "data" / "external"
PIO = ROOT / "_pio_yolo"
OUT_DIR = ROOT / "runs_external_eval_occ" / "preview_overlay"

IMGSZ = 960
FALLBACK_IMGSZ = 640
CONF = 0.25
MAX_DET = 1000                 # lift the default 300 cap for dense scenes
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
N_SAMPLE = 8
SEED = 42
BROILER_VAL_FRAC = 0.20
GT_COLOR = (0, 200, 0)         # BGR green
PRED_COLOR = (255, 60, 0)      # BGR blue


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


def parse_gt(txt_path, w, h):
    boxes = []
    if not txt_path.exists():
        return boxes
    for line in txt_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        coords = list(map(float, parts[1:]))
        if len(coords) == 4:
            xc, yc, bw, bh = coords
            x1, y1 = (xc - bw / 2) * w, (yc - bh / 2) * h
            x2, y2 = (xc + bw / 2) * w, (yc + bh / 2) * h
        else:
            xs, ys = coords[0::2], coords[1::2]
            x1, x2 = min(xs) * w, max(xs) * w
            y1, y2 = min(ys) * h, max(ys) * h
        boxes.append((int(x1), int(y1), int(x2), int(y2)))
    return boxes


def predict_boxes(model, img):
    try:
        r = model.predict(source=str(img), imgsz=IMGSZ, conf=CONF, batch=1,
                          max_det=MAX_DET, save=False, verbose=False)[0]
    except RuntimeError as e:
        if "memory" not in str(e).lower():
            raise
        r = model.predict(source=str(img), imgsz=FALLBACK_IMGSZ, conf=CONF,
                          batch=1, max_det=MAX_DET, save=False, verbose=False)[0]
    return [tuple(map(int, b)) for b in r.boxes.xyxy.tolist()]


def render(model, ds_name, imgs, label_dir):
    sample = evenly_spaced(imgs, N_SAMPLE)
    out = OUT_DIR / ds_name
    out.mkdir(parents=True, exist_ok=True)
    log(f"[{ds_name}] overlay {len(sample)} imgs -> {out}")
    for img in sample:
        im = cv2.imread(str(img))
        if im is None:
            log(f"  !! cannot read {img.name}")
            continue
        h, w = im.shape[:2]
        gt = parse_gt(label_dir / f"{img.stem}.txt", w, h)
        pred = predict_boxes(model, img)
        # GT first (green), predictions on top (blue)
        for (x1, y1, x2, y2) in gt:
            cv2.rectangle(im, (x1, y1), (x2, y2), GT_COLOR, 2)
        for (x1, y1, x2, y2) in pred:
            cv2.rectangle(im, (x1, y1), (x2, y2), PRED_COLOR, 2)
        # legend
        cv2.rectangle(im, (0, 0), (430, 40), (0, 0, 0), -1)
        cv2.putText(im, f"GT={len(gt)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, GT_COLOR, 2, cv2.LINE_AA)
        cv2.putText(im, f"PRED={len(pred)}", (200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, PRED_COLOR, 2, cv2.LINE_AA)
        dst = out / f"{img.stem}__gt{len(gt)}_pred{len(pred)}.jpg"
        cv2.imwrite(str(dst), im)
        log(f"  {img.name}: GT={len(gt)} PRED={len(pred)} -> {dst.name}")
    log(f"[{ds_name}] done\n")


def main():
    log(f"Model: {MODEL_PATH}")
    assert MODEL_PATH.exists(), f"weights not found: {MODEL_PATH}"
    model = YOLO(str(MODEL_PATH))
    log(f"Model classes: {model.names}, conf={CONF}, imgsz={IMGSZ}, "
        f"max_det={MAX_DET}\n")
    log("Legend: GREEN=ground truth, BLUE=prediction\n")
    for ds_name, (imgs, label_dir) in sources().items():
        render(model, ds_name, imgs, label_dir)
    log(f"ALL DONE -> {OUT_DIR}")


if __name__ == "__main__":
    main()
