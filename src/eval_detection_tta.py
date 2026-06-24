"""
eval_detection_tta.py — Evaluasi deteksi YOLO (mAP) dengan Test-Time Augmentation (TTA).

Tujuan:
  Cek "gratis" apakah TTA (multi-scale + horizontal flip, digabung sebelum NMS)
  menaikkan mAP tanpa perlu melatih ulang model. Ultralytics mode `val` biasa TIDAK
  mengaktifkan TTA, jadi di sini kita paksa `augment=True` saat validasi.

  Skema output JSON/CSV SENGAJA dibuat identik dengan eval_detection.py sehingga
  hasilnya bisa langsung dibandingkan oleh src/compare_ab.py (mis. TTA vs baseline).

Reuse dari eval_detection.py (tidak didefinisikan ulang):
  DATASETS, resolve_val_dirs(), count_images()

Metode:
  Jalur utama = model.val(data=..., augment=True) yang di versi Ultralytics terbaru
  memang menghitung metrik lengkap dengan TTA (mAP50/50-95, P, R) — identik cara baca
  metriknya dengan eval_detection.py.evaluate_one (metrics.box.map50/map/mp/mr).
  Jika versi Ultralytics terpasang menolak `augment=True` saat val, dataset itu
  ditandai status "tta_unsupported" (bukan gagal total) agar dataset lain tetap jalan.

Kondisi:
  --rectified-root tidak diisi -> condition = "tta"       (gambar asli + TTA)
  --rectified-root diisi       -> condition = "mowa_tta"  (gambar MOWA rectified + TTA)

Contoh:
  # TTA di atas gambar asli (bandingkan dengan reports/eval_baseline.json)
  .venv-yolo/Scripts/python.exe src/eval_detection_tta.py \
      --weights "train model/runs_compare/cmp_yolov8m/weights/best.pt" \
      --out reports/eval_tta.json

  # TTA di atas gambar hasil MOWA (bandingkan dengan reports/eval_mowa.json)
  .venv-yolo/Scripts/python.exe src/eval_detection_tta.py \
      --weights "train model/runs_compare/cmp_yolov8m/weights/best.pt" \
      --rectified-root data/rectified \
      --out reports/eval_mowa_tta.json

  # lalu diff seperti biasa
  .venv-yolo/Scripts/python.exe src/compare_ab.py \
      --baseline reports/eval_baseline.json --mowa reports/eval_tta.json
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

# Reuse definisi dataset & helper dari modul non-TTA agar tetap satu sumber kebenaran.
from eval_detection import DATASETS, count_images, resolve_val_dirs

ROOT = Path(__file__).resolve().parents[1]


def evaluate_one_tta(weights: Path, ds: Dict, val_dirs: List[Path], imgsz: int, device: str) -> Dict:
    """Sama seperti eval_detection.evaluate_one, tetapi val dengan augment=True (TTA).

    Multi-scale + horizontal flip digabung sebelum NMS oleh Ultralytics ketika
    augment=True. Bila versi Ultralytics tidak mendukungnya saat `val`, kembalikan
    status "tta_unsupported" (dataset lain tetap dievaluasi).
    """
    from ultralytics import YOLO

    existing = [d for d in val_dirs if d.is_dir() and count_images([d]) > 0]
    if not existing:
        return {
            "id": ds["id"], "display": ds["display"], "in_domain": ds["in_domain"],
            "status": "no_images", "images": 0, "val_dirs": [str(d) for d in val_dirs],
        }

    n_imgs = count_images(existing)
    # data.yaml sementara. names 1 kelas (0='pollo') agar cocok model single-class.
    # `train` wajib ada walau hanya val; arahkan ke folder val yang sama.
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

    try:
        model = YOLO(str(weights))
        metrics = model.val(
            data=tmp_yaml,
            imgsz=imgsz,
            device=device,
            augment=True,  # <- inilah TTA (multi-scale + flip, merge pre-NMS)
            verbose=False,
            save_json=False,
            plots=False,
            project=str(ROOT / "reports" / "_ultra_val_tta"),
            name=ds["id"],
            exist_ok=True,
        )
    except (TypeError, ValueError, KeyError, SyntaxError) as exc:
        # HANYA menangani kasus "argumen augment ditolak / tidak dikenali" oleh versi
        # Ultralytics ini (memunculkan TypeError/ValueError/KeyError/SyntaxError saat
        # cek konfigurasi). Kegagalan runtime asli (CUDA OOM -> RuntimeError, bobot
        # rusak, gambar tak terbaca) SENGAJA tidak ditangkap agar gagal keras, bukan
        # tersamar jadi "tta_unsupported" yang bisa membelokkan verdict A/B.
        print(f"   TTA tidak didukung versi Ultralytics ini untuk dataset ini: {exc}", file=sys.stderr)
        return {
            "id": ds["id"], "display": ds["display"], "in_domain": ds["in_domain"],
            "status": "tta_unsupported", "images": n_imgs,
            "val_dirs": [str(d) for d in existing],
        }
    finally:
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
    ap = argparse.ArgumentParser(description="Evaluasi mAP YOLO dengan TTA (multi-scale + flip).")
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="Path JSON hasil (mis. reports/eval_tta.json).")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default="0", help="'0' untuk GPU cuda:0, 'cpu' untuk CPU.")
    ap.add_argument("--rectified-root", type=Path, default=None,
                    help="Jika diisi (mis. data/rectified), evaluasi gambar hasil MOWA di <root>/<id>/images.")
    ap.add_argument("--only", default="", help="Evaluasi hanya dataset id ini (opsional).")
    args = ap.parse_args()

    if not args.weights.exists():
        print(f"ERROR: weights tidak ditemukan: {args.weights}", file=sys.stderr)
        return 2

    condition = "mowa_tta" if args.rectified_root else "tta"
    targets = [d for d in DATASETS if not args.only or d["id"] == args.only]
    results = []
    for ds in targets:
        val_dirs = resolve_val_dirs(ds, args.rectified_root)
        print(f"[eval-tta] {condition} :: {ds['id']} <- {[str(d) for d in val_dirs]}")
        res = evaluate_one_tta(args.weights, ds, val_dirs, args.imgsz, args.device)
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

    # CSV pendamping (skema sama seperti eval_detection.py).
    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", "dataset", "in_domain", "images", "map50", "map50_95", "precision", "recall", "status"])
        for r in results:
            w.writerow([condition, r["id"], r.get("in_domain"), r.get("images", 0),
                        r.get("map50", ""), r.get("map50_95", ""), r.get("precision", ""),
                        r.get("recall", ""), r.get("status", "")])

    print(f"[eval-tta] tulis {args.out}")
    print(f"[eval-tta] tulis {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
