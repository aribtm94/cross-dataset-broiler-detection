"""Verification: isolate MODEL vs SPLIT for the surprising FUM val-only jump.

Baseline (summary.json top-level): yolo11m on fum FULL (train+valid+test, 326) -> mAP50 0.139
New (summary.json val_only):       yolo8m  on fum VALID only (18)            -> mAP50 0.842

The jump conflates two changes (model AND scope). Run BOTH models on the SAME
18-image valid split so the only variable is the model. If yolo11m ALSO scores
high on the 18 valid imgs, the driver is the split (valid subset is easy /
better-annotated), not the yolo11m->yolo8m switch.

Cheap: 18 imgs, val only, batch=1, imgsz=960. Sequential (RAM constraint).
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
EXTERNAL_DIR = ROOT.parent / "data" / "data" / "external"
TMP_YAML_DIR = ROOT / "tmp_data_yaml"
OUT_DIR = ROOT / "runs_external_eval"
IMGSZ = 960

MODELS = {
    "yolov8m": ROOT / "runs_compare" / "cmp_yolov8m" / "weights" / "best.pt",
    "yolo11m": ROOT / "runs_compare" / "cmp_yolo11m" / "weights" / "best.pt",
}

# same yaml as the val_only run for fum (valid/ split only)
FUM_YAML = {"path": str((EXTERNAL_DIR / "chicken_detection_fum").resolve()),
            "train": "train/images", "val": "valid/images",
            "nc": 1, "names": ["chicken"]}


def main():
    yaml_path = TMP_YAML_DIR / "fum_valonly_verify.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(FUM_YAML, f, sort_keys=False)

    out = {}
    for tag, mp in MODELS.items():
        log(f"========== {tag} on FUM valid-only ==========")
        model = YOLO(str(mp))
        m = model.val(data=str(yaml_path), split="val", imgsz=IMGSZ, batch=1,
                      project=str(OUT_DIR / "chicken_detection_fum"),
                      name=f"valverify_{tag}", exist_ok=True, verbose=False)
        res = {"precision": float(m.box.mp), "recall": float(m.box.mr),
               "map50": float(m.box.map50), "map50_95": float(m.box.map)}
        out[tag] = res
        log(f"[{tag}] P={res['precision']:.3f} R={res['recall']:.3f} "
            f"mAP50={res['map50']:.3f} mAP50-95={res['map50_95']:.3f}")

    log("\n========== FUM valid-only: model comparison ==========")
    log(json.dumps(out, indent=2))
    (OUT_DIR / "fum_valonly_verify.json").write_text(json.dumps(out, indent=2))
    log("ALL DONE")


if __name__ == "__main__":
    main()
