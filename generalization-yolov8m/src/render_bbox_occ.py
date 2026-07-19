"""Render bounding-box predictions of the occlusion-aug YOLOv8m model.

Saves annotated images (predicted boxes drawn) for a sample of VALIDATION
images from each of the 3 datasets, using the same val sources as the eval.
Output: runs_external_eval_occ/preview/<dataset>/*.jpg
"""
import random
from pathlib import Path

import torch
torch.set_num_threads(2)

from ultralytics import YOLO


def log(m):
    print(m, flush=True)


ROOT = Path(__file__).resolve().parent
MODEL_PATH = (ROOT / "cmp_yolov8m_occ_rect" / "cmp_yolov8m_occ_rect"
              / "weights" / "best.pt")
EXTERNAL_DIR = ROOT.parent / "data" / "data" / "external"
PIO = ROOT / "_pio_yolo"
OUT_DIR = ROOT / "runs_external_eval_occ" / "preview"

IMGSZ = 960
FALLBACK_IMGSZ = 640
CONF = 0.25
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
N_SAMPLE = 8          # images to render per dataset
SEED = 42
BROILER_VAL_FRAC = 0.20


def list_images(d):
    return [p for p in sorted(Path(d).iterdir()) if p.suffix.lower() in IMG_EXTS]


def evenly_spaced(items, k):
    """Pick k items spread across the list (deterministic, coverage-friendly)."""
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
        "pio_original_val": list_images(PIO / "images" / "val"),
        "broiler_instance_seg": broiler_val_imgs(),
        "chicken_detection_fum": list_images(
            EXTERNAL_DIR / "chicken_detection_fum" / "valid" / "images"),
    }


def render(model, ds_name, imgs):
    sample = evenly_spaced(imgs, N_SAMPLE)
    out = OUT_DIR / ds_name
    out.mkdir(parents=True, exist_ok=True)
    log(f"[{ds_name}] rendering {len(sample)} imgs -> {out}")
    total_det = 0
    for img in sample:
        try:
            r = model.predict(source=str(img), imgsz=IMGSZ, conf=CONF, batch=1,
                              save=False, verbose=False)[0]
        except RuntimeError as e:
            if "memory" not in str(e).lower():
                raise
            r = model.predict(source=str(img), imgsz=FALLBACK_IMGSZ, conf=CONF,
                              batch=1, save=False, verbose=False)[0]
        n = len(r.boxes)
        total_det += n
        # draw boxes and write to our own path (avoids ultralytics run-dir sprawl)
        annotated = r.plot(line_width=2)  # BGR ndarray
        import cv2
        dst = out / f"{img.stem}__det{n}.jpg"
        cv2.imwrite(str(dst), annotated)
        log(f"  {img.name}: {n} boxes -> {dst.name}")
    log(f"[{ds_name}] done, {total_det} boxes over {len(sample)} imgs\n")


def main():
    log(f"Model: {MODEL_PATH}")
    assert MODEL_PATH.exists(), f"weights not found: {MODEL_PATH}"
    model = YOLO(str(MODEL_PATH))
    log(f"Model classes: {model.names}, conf={CONF}, imgsz={IMGSZ}\n")
    for ds_name, imgs in sources().items():
        render(model, ds_name, imgs)
    log(f"ALL DONE -> {OUT_DIR}")


if __name__ == "__main__":
    main()
