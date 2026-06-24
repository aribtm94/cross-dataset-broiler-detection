"""
eval_detection.py — Evaluasi deteksi YOLO (mAP) pada beberapa dataset sekaligus.

Dipakai untuk dua kondisi eksperimen A/B skripsi:
  A. BASELINE       : gambar asli               (--rectified-root tidak diisi)
  B. MOWA-rectified : gambar hasil MOWA + label warp (--rectified-root data/rectified)

Model yang sama (YOLOv8m best.pt hasil training PIO) dievaluasi pada:
  - PIO val               (data/images/val            + data/labels/val)
  - broiler_instance_seg  (Roboflow, single class)
  - chicken_detection_fum (Roboflow, single class, 3 split digabung)

Ultralytics menurunkan folder label otomatis dengan mengganti '/images/' -> '/labels/'
pada path gambar, sehingga tidak perlu menyalin apa pun; cukup arahkan `val` ke folder
images tiap dataset. Semua dataset single-class (id kelas 0), jadi cocok dengan model
yang juga single-class (0='pollo').

Contoh:
  # Kondisi A (baseline)
  .venv-yolo/Scripts/python.exe src/eval_detection.py \
      --weights "train model/runs_compare/cmp_yolov8m/weights/best.pt" \
      --out reports/eval_baseline.json

  # Kondisi B (MOWA) — setelah rectify+warp mengisi data/rectified/<id>/{images,labels}
  .venv-yolo/Scripts/python.exe src/eval_detection.py \
      --weights "train model/runs_compare/cmp_yolov8m/weights/best.pt" \
      --rectified-root data/rectified \
      --out reports/eval_mowa.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import yaml

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Definisi dataset. `val_dirs` = daftar folder images (boleh >1 split) untuk BASELINE.
# Untuk kondisi MOWA, gambar dibaca dari <rectified_root>/<rectified_subdir>/images.
DATASETS = [
    {
        "id": "pio_val",
        "display": "PIO val (in-domain)",
        "val_dirs": [ROOT / "data" / "images" / "val"],
        "rectified_subdir": "pio_val",
        "in_domain": True,
    },
    {
        "id": "broiler_instance_seg",
        "display": "Roboflow broiler_instance_seg (external)",
        "val_dirs": [ROOT / "data" / "external" / "broiler_instance_seg" / "train" / "images"],
        "rectified_subdir": "broiler_instance_seg",
        "in_domain": False,
    },
    {
        "id": "chicken_detection_fum",
        "display": "Roboflow chicken_detection_fum (external)",
        "val_dirs": [
            ROOT / "data" / "external" / "chicken_detection_fum" / "test" / "images",
            ROOT / "data" / "external" / "chicken_detection_fum" / "valid" / "images",
            ROOT / "data" / "external" / "chicken_detection_fum" / "train" / "images",
        ],
        "rectified_subdir": "chicken_detection_fum",
        "in_domain": False,
    },
]


def count_images(dirs: List[Path]) -> int:
    n = 0
    for d in dirs:
        if d.is_dir():
            n += sum(1 for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    return n


def resolve_val_dirs(ds: Dict, rectified_root: Path | None) -> List[Path]:
    """Folder images yang dievaluasi. Baseline -> val_dirs asli; MOWA -> <root>/<sub>/images."""
    if rectified_root is None:
        return [Path(d) for d in ds["val_dirs"]]
    rect = rectified_root / ds["rectified_subdir"] / "images"
    return [rect]


def evaluate_one(weights: Path, ds: Dict, val_dirs: List[Path], imgsz: int, device: str) -> Dict:
    """Jalankan Ultralytics val untuk satu dataset, kembalikan metrik ringkas."""
    from ultralytics import YOLO

    existing = [d for d in val_dirs if d.is_dir() and count_images([d]) > 0]
    if not existing:
        return {
            "id": ds["id"], "display": ds["display"], "in_domain": ds["in_domain"],
            "status": "no_images", "images": 0, "val_dirs": [str(d) for d in val_dirs],
        }

    n_imgs = count_images(existing)
    # data.yaml sementara. names dibuat 1 kelas agar cocok dengan model single-class.
    # `train` wajib ada di data.yaml Ultralytics walau kita hanya val; arahkan ke folder
    # val yang sama (tidak dipakai untuk training, hanya lolos validasi schema).
    data_cfg = {
        "path": str(ROOT),
        "train": [str(d) for d in existing],
        "val": [str(d) for d in existing],
        "nc": 1,
        "names": {0: "pollo"},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as tf:
        yaml.safe_dump(data_cfg, tf, allow_unicode=True, sort_keys=False)
        tmp_yaml = tf.name

    model = YOLO(str(weights))
    metrics = model.val(
        data=tmp_yaml,
        imgsz=imgsz,
        device=device,
        verbose=False,
        save_json=False,
        plots=False,
        project=str(ROOT / "reports" / "_ultra_val"),
        name=ds["id"],
        exist_ok=True,
    )
    Path(tmp_yaml).unlink(missing_ok=True)

    box = metrics.box
    return {
        "id": ds["id"],
        "display": ds["display"],
        "in_domain": ds["in_domain"],
        "status": "ok",
        "images": n_imgs,
        "val_dirs": [str(d) for d in existing],
        "map50": round(float(box.map50), 5),
        "map50_95": round(float(box.map), 5),
        "precision": round(float(box.mp), 5),
        "recall": round(float(box.mr), 5),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluasi mAP YOLO pada PIO + external.")
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="Path JSON hasil (mis. reports/eval_baseline.json).")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default="0", help="'0' untuk GPU cuda:0, 'cpu' untuk CPU.")
    ap.add_argument("--rectified-root", type=Path, default=None,
                    help="Jika diisi (mis. data/rectified), evaluasi gambar hasil MOWA di <root>/<id>/images.")
    ap.add_argument("--only", default="", help="Evaluasi hanya dataset id ini (opsional).")
    args = ap.parse_args()

    if not args.weights.exists():
        print(f"ERROR: weights tidak ditemukan: {args.weights}", file=sys.stderr)
        return 2

    condition = "mowa_rectified" if args.rectified_root else "baseline"
    targets = [d for d in DATASETS if not args.only or d["id"] == args.only]
    results = []
    for ds in targets:
        val_dirs = resolve_val_dirs(ds, args.rectified_root)
        print(f"[eval] {condition} :: {ds['id']} <- {[str(d) for d in val_dirs]}")
        res = evaluate_one(args.weights, ds, val_dirs, args.imgsz, args.device)
        status = res.get("status")
        if status == "ok":
            print(f"   mAP50={res['map50']:.4f}  mAP50-95={res['map50_95']:.4f} "
                  f"P={res['precision']:.4f} R={res['recall']:.4f}  (n={res['images']})")
        else:
            print(f"   SKIP ({status})")
        results.append(res)

    payload = {
        "condition": condition,
        "weights": str(args.weights),
        "imgsz": args.imgsz,
        "rectified_root": str(args.rectified_root) if args.rectified_root else None,
        "datasets": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # CSV pendamping untuk dibaca cepat / dashboard.
    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", "dataset", "in_domain", "images", "map50", "map50_95", "precision", "recall", "status"])
        for r in results:
            w.writerow([condition, r["id"], r.get("in_domain"), r.get("images", 0),
                        r.get("map50", ""), r.get("map50_95", ""), r.get("precision", ""),
                        r.get("recall", ""), r.get("status", "")])

    print(f"[eval] tulis {args.out}")
    print(f"[eval] tulis {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
