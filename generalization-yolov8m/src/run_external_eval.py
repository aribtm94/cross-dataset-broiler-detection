"""Run predict + validate with best.pt over the 3 external datasets.

Low-memory host (only ~3.5 GB free): run at the training imgsz (960) but with
batch=1 and streaming so peak memory stays ~2-3 GB. Per-image fallback to 640 on
OOM keeps a long run from dying on a transient memory dip.
"""
import json
from pathlib import Path

import torch
torch.set_num_threads(2)  # cap CPU memory/threads on a low-RAM host

import yaml
from ultralytics import YOLO


def log(msg):
    print(msg, flush=True)


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "runs_compare" / "cmp_yolo11m" / "weights" / "best.pt"
EXTERNAL_DIR = ROOT.parent / "data" / "data" / "external"
OUT_DIR = ROOT / "runs_external_eval"
TMP_YAML_DIR = ROOT / "tmp_data_yaml"
TMP_YAML_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

IMGSZ = 960          # training image size -> fair validation metrics
FALLBACK_IMGSZ = 640  # used per-image only if 960 OOMs
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DATASETS = {
    "broiler_instance_seg": {
        "predict_sources": ["train/images"],
        "val_yaml": {
            "path": str((EXTERNAL_DIR / "broiler_instance_seg").resolve()),
            "train": "train/images",
            "val": "train/images",
            "nc": 1,
            "names": ["chicken"],
        },
    },
    "chicken_detection_fum": {
        "predict_sources": ["train/images", "valid/images", "test/images"],
        "val_yaml": {
            "path": str((EXTERNAL_DIR / "chicken_detection_fum").resolve()),
            "train": "train/images",
            "val": ["train/images", "valid/images", "test/images"],
            "nc": 1,
            "names": ["chicken"],
        },
    },
    "nestler_yolo": {
        "predict_sources": ["images/val"],
        "val_yaml": {
            "path": str((EXTERNAL_DIR / "nestler_yolo").resolve()),
            "train": "images/val",
            "val": "images/val",
            "nc": 1,
            "names": ["chicken"],
        },
    },
}


def predict_one(model, img_path, ds_name):
    """Predict a single image; fall back to a smaller imgsz on OOM."""
    try:
        return model.predict(
            source=img_path, imgsz=IMGSZ, batch=1,
            save=True, save_txt=True, save_conf=True,
            project=str(OUT_DIR / ds_name), name="predict",
            exist_ok=True, verbose=False,
        )[0]
    except RuntimeError as e:
        if "memory" not in str(e).lower():
            raise
        log(f"[{ds_name}] OOM at imgsz={IMGSZ} on {Path(img_path).name}, retrying at {FALLBACK_IMGSZ}")
        return model.predict(
            source=img_path, imgsz=FALLBACK_IMGSZ, batch=1,
            save=True, save_txt=True, save_conf=True,
            project=str(OUT_DIR / ds_name), name="predict",
            exist_ok=True, verbose=False,
        )[0]


def run_predict(model, ds_name, ds_root, sources):
    files = []
    for s in sources:
        d = ds_root / s
        if d.exists():
            files.extend(str(p) for p in sorted(d.iterdir()) if p.suffix.lower() in IMG_EXTS)
    total = len(files)
    log(f"[{ds_name}] PREDICT start: {total} images (imgsz={IMGSZ})")

    n_det = 0
    for done, img in enumerate(files, 1):
        r = predict_one(model, img, ds_name)
        n_det += len(r.boxes)
        if done % 20 == 0 or done == total:
            pct = 100.0 * done / total if total else 100.0
            log(f"[{ds_name}] PREDICT {done}/{total} ({pct:.0f}%) - detections so far: {n_det}")
    log(f"[{ds_name}] PREDICT done: {total} images, {n_det} total detections")
    return total, n_det


def run_val(model, ds_name, cfg):
    yaml_path = TMP_YAML_DIR / f"{ds_name}.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(cfg["val_yaml"], f, sort_keys=False)
    log(f"[{ds_name}] VALIDATE start (imgsz={IMGSZ}, batch=1)")
    metrics = model.val(
        data=str(yaml_path), split="val", imgsz=IMGSZ, batch=1,
        project=str(OUT_DIR / ds_name), name="val", exist_ok=True, verbose=False,
    )
    res = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }
    log(f"[{ds_name}] VALIDATE done: P={res['precision']:.3f} R={res['recall']:.3f} "
        f"mAP50={res['map50']:.3f} mAP50-95={res['map50_95']:.3f}")
    return res


def main():
    log(f"Model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    log(f"Model classes: {model.names}")
    log(f"Datasets: {list(DATASETS)}\n")

    summary = {}
    n_ds = len(DATASETS)
    for i, (ds_name, cfg) in enumerate(DATASETS.items(), 1):
        ds_root = EXTERNAL_DIR / ds_name
        log(f"========== [{i}/{n_ds}] {ds_name} ==========")
        n_images, n_det = run_predict(model, ds_name, ds_root, cfg["predict_sources"])
        val_res = run_val(model, ds_name, cfg)
        summary[ds_name] = {"n_images": n_images, "n_detections": n_det, **val_res}
        with open(OUT_DIR / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        log("")

    log("========== SUMMARY ==========")
    log(json.dumps(summary, indent=2))
    log(f"\nSaved: {OUT_DIR / 'summary.json'}")
    log("ALL DONE")


if __name__ == "__main__":
    main()
