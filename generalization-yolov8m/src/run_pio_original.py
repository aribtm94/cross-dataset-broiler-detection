"""Baseline: run best.pt on the PIO original val split (in-domain reference).

Same low-memory settings as run_external_eval.py (imgsz=960, batch=1).
Appends result to runs_external_eval/summary.json under key 'pio_original_val'.
"""
import json
from pathlib import Path

import torch
torch.set_num_threads(2)

import yaml
from ultralytics import YOLO


def log(m):
    print(m, flush=True)


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "runs_compare" / "cmp_yolo11m" / "weights" / "best.pt"
PIO = ROOT / "_pio_yolo"
OUT_DIR = ROOT / "runs_external_eval"
TMP_YAML_DIR = ROOT / "tmp_data_yaml"
IMGSZ = 960
FALLBACK = 640
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NAME = "pio_original_val"


def predict_one(model, img):
    try:
        return model.predict(source=img, imgsz=IMGSZ, batch=1, save=True, save_txt=True,
                             save_conf=True, project=str(OUT_DIR / NAME), name="predict",
                             exist_ok=True, verbose=False)[0]
    except RuntimeError as e:
        if "memory" not in str(e).lower():
            raise
        log(f"OOM at 960 on {Path(img).name}, retry {FALLBACK}")
        return model.predict(source=img, imgsz=FALLBACK, batch=1, save=True, save_txt=True,
                             save_conf=True, project=str(OUT_DIR / NAME), name="predict",
                             exist_ok=True, verbose=False)[0]


def main():
    model = YOLO(str(MODEL_PATH))
    log(f"Model: {MODEL_PATH} classes={model.names}")

    val_dir = PIO / "images" / "val"
    files = [str(p) for p in sorted(val_dir.iterdir()) if p.suffix.lower() in IMG_EXTS]
    total = len(files)
    log(f"[{NAME}] PREDICT start: {total} images (imgsz={IMGSZ})")
    n_det = 0
    for done, img in enumerate(files, 1):
        r = predict_one(model, img)
        n_det += len(r.boxes)
        if done % 25 == 0 or done == total:
            log(f"[{NAME}] PREDICT {done}/{total} ({100*done/total:.0f}%) - detections: {n_det}")
    log(f"[{NAME}] PREDICT done: {total} images, {n_det} detections")

    # validate against local path (override the absolute path baked into the training dataset.yaml)
    yaml_path = TMP_YAML_DIR / f"{NAME}.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump({"path": str(PIO.resolve()), "train": "images/train",
                        "val": "images/val", "nc": 1, "names": ["chicken"]}, f, sort_keys=False)
    log(f"[{NAME}] VALIDATE start (imgsz={IMGSZ}, batch=1)")
    m = model.val(data=str(yaml_path), split="val", imgsz=IMGSZ, batch=1,
                  project=str(OUT_DIR / NAME), name="val", exist_ok=True, verbose=False)
    res = {"n_images": total, "n_detections": n_det,
           "precision": float(m.box.mp), "recall": float(m.box.mr),
           "map50": float(m.box.map50), "map50_95": float(m.box.map)}
    log(f"[{NAME}] VALIDATE done: P={res['precision']:.3f} R={res['recall']:.3f} "
        f"mAP50={res['map50']:.3f} mAP50-95={res['map50_95']:.3f}")

    sp = OUT_DIR / "summary.json"
    summary = json.loads(sp.read_text()) if sp.exists() else {}
    # put baseline first
    summary = {NAME: res, **{k: v for k, v in summary.items() if k != NAME}}
    sp.write_text(json.dumps(summary, indent=2))
    log(f"Updated {sp}")
    log("ALL DONE")


if __name__ == "__main__":
    main()
