"""Val-only re-test with the OCCLUSION-AUGMENTED YOLOv8m model.

Same methodology as run_val_only_eval.py, but evaluates the new occlusion-aug
weights (cmp_yolov8m_occ_rect, trained with erasing=0.4 + mixup=0.1, imgsz=960)
over the 3 non-nestler datasets on the VALIDATION split only. Results are
written to runs_external_eval/summary.json under the "val_only_occ" key so the
existing "val_only" (plain yolov8m) block stays intact for A/B comparison.

Per-dataset val source (identical to the baseline script):
  - pio_original_val      -> _pio_yolo/images/val            (452 real val imgs)
  - broiler_instance_seg  -> deterministic 20% of train/images as val
                             (seed=42; no valid/ folder ships with the dataset)
  - chicken_detection_fum -> valid/images                    (18 real val imgs)

Low-memory host: imgsz=960 (match training) but batch=1 + per-image predict.
"""
import json
import random
from pathlib import Path

import torch
torch.set_num_threads(2)  # cap CPU threads/memory on a low-RAM host

import yaml
from ultralytics import YOLO


def log(m):
    print(m, flush=True)


ROOT = Path(__file__).resolve().parent
MODEL_PATH = (ROOT / "cmp_yolov8m_occ_rect" / "cmp_yolov8m_occ_rect"
              / "weights" / "best.pt")
EXTERNAL_DIR = ROOT.parent / "data" / "data" / "external"
PIO = ROOT / "_pio_yolo"
OUT_DIR = ROOT / "runs_external_eval_occ"
SUMMARY_PATH = ROOT / "runs_external_eval" / "summary.json"
TMP_YAML_DIR = ROOT / "tmp_data_yaml"
TMP_YAML_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

IMGSZ = 960
FALLBACK_IMGSZ = 640
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BROILER_VAL_FRAC = 0.20
BROILER_SEED = 42
SUMMARY_KEY = "val_only_occ"


def list_images(d):
    return [p for p in sorted(Path(d).iterdir()) if p.suffix.lower() in IMG_EXTS]


def make_broiler_val_split():
    """Deterministic 20% of broiler train/images -> a val list .txt (reproducible).

    Uses the SAME seed/frac as the baseline script so the val set is identical.
    """
    train_img_dir = EXTERNAL_DIR / "broiler_instance_seg" / "train" / "images"
    imgs = list_images(train_img_dir)
    rng = random.Random(BROILER_SEED)
    shuffled = imgs[:]
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * BROILER_VAL_FRAC))
    val_imgs = sorted(shuffled[:n_val], key=lambda p: p.name)
    txt = TMP_YAML_DIR / "broiler_val_split.txt"
    txt.write_text("\n".join(str(p.resolve()) for p in val_imgs) + "\n")
    log(f"[broiler] built val split: {n_val}/{len(imgs)} imgs "
        f"(seed={BROILER_SEED}, frac={BROILER_VAL_FRAC}) -> {txt.name}")
    return val_imgs, txt


def build_datasets():
    ds = {}

    pio_val = list_images(PIO / "images" / "val")
    ds["pio_original_val"] = {
        "images": pio_val,
        "yaml": {"path": str(PIO.resolve()), "train": "images/train",
                 "val": "images/val", "nc": 1, "names": ["chicken"]},
        "note": "real val split (452 imgs)",
    }

    broiler_val, broiler_txt = make_broiler_val_split()
    ds["broiler_instance_seg"] = {
        "images": broiler_val,
        "yaml": {"path": str((EXTERNAL_DIR / "broiler_instance_seg").resolve()),
                 "train": "train/images", "val": str(broiler_txt.resolve()),
                 "nc": 1, "names": ["chicken"]},
        "note": f"NO val folder in dataset; synthetic {int(BROILER_VAL_FRAC*100)}% "
                f"of train as val (seed={BROILER_SEED})",
    }

    fum_val = list_images(EXTERNAL_DIR / "chicken_detection_fum" / "valid" / "images")
    ds["chicken_detection_fum"] = {
        "images": fum_val,
        "yaml": {"path": str((EXTERNAL_DIR / "chicken_detection_fum").resolve()),
                 "train": "train/images", "val": "valid/images",
                 "nc": 1, "names": ["chicken"]},
        "note": "real valid/ split only (18 imgs)",
    }
    return ds


def predict_count(model, imgs, ds_name):
    """Count detections over the val images (no disk save). Per-image, batch=1."""
    total = len(imgs)
    log(f"[{ds_name}] PREDICT(count) start: {total} imgs (imgsz={IMGSZ})")
    n_det = 0
    for done, img in enumerate(imgs, 1):
        try:
            r = model.predict(source=str(img), imgsz=IMGSZ, batch=1,
                              save=False, verbose=False)[0]
        except RuntimeError as e:
            if "memory" not in str(e).lower():
                raise
            log(f"[{ds_name}] OOM at {IMGSZ} on {img.name}, retry {FALLBACK_IMGSZ}")
            r = model.predict(source=str(img), imgsz=FALLBACK_IMGSZ, batch=1,
                              save=False, verbose=False)[0]
        n_det += len(r.boxes)
        if done % 25 == 0 or done == total:
            log(f"[{ds_name}] PREDICT {done}/{total} - det so far: {n_det}")
    log(f"[{ds_name}] PREDICT done: {n_det} detections")
    return n_det


def run_val(model, ds_name, cfg):
    yaml_path = TMP_YAML_DIR / f"{ds_name}__valonly_occ.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(cfg["yaml"], f, sort_keys=False)
    log(f"[{ds_name}] VALIDATE start (imgsz={IMGSZ}, batch=1)")
    m = model.val(data=str(yaml_path), split="val", imgsz=IMGSZ, batch=1,
                  project=str(OUT_DIR / ds_name), name="val_only", exist_ok=True,
                  verbose=False)
    res = {"precision": float(m.box.mp), "recall": float(m.box.mr),
           "map50": float(m.box.map50), "map50_95": float(m.box.map)}
    log(f"[{ds_name}] VALIDATE done: P={res['precision']:.3f} R={res['recall']:.3f} "
        f"mAP50={res['map50']:.3f} mAP50-95={res['map50_95']:.3f}")
    return res


def main():
    log(f"Model: {MODEL_PATH}")
    assert MODEL_PATH.exists(), f"weights not found: {MODEL_PATH}"
    model = YOLO(str(MODEL_PATH))
    log(f"Model classes: {model.names}\n")

    datasets = build_datasets()
    val_block = {
        "_note": ("Val-split-only re-test (except nestler). broiler has no valid/ "
                  "folder so a deterministic 20% train-as-val split (seed=42) is "
                  "used. imgsz=960, batch=1."),
        "_model": ("yolov8m + occlusion aug (erasing=0.4, mixup=0.1) -- "
                   "cmp_yolov8m_occ_rect/weights/best.pt. Compare vs the "
                   "'val_only' block (plain yolov8m)."),
    }

    for i, (ds_name, cfg) in enumerate(datasets.items(), 1):
        log(f"========== [{i}/{len(datasets)}] {ds_name} (val only, occ) ==========")
        n_images = len(cfg["images"])
        n_det = predict_count(model, cfg["images"], ds_name)
        val_res = run_val(model, ds_name, cfg)
        val_block[ds_name] = {"n_images": n_images, "n_detections": n_det,
                              "val_source_note": cfg["note"], **val_res}
        # persist incrementally so a crash keeps finished datasets
        summary = json.loads(SUMMARY_PATH.read_text()) if SUMMARY_PATH.exists() else {}
        summary[SUMMARY_KEY] = val_block
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
        log(f"[{ds_name}] written to summary.json[{SUMMARY_KEY}]\n")

    log("========== VAL_ONLY_OCC SUMMARY ==========")
    log(json.dumps(val_block, indent=2))
    log("ALL DONE")


if __name__ == "__main__":
    main()
