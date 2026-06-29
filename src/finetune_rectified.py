"""
finetune_rectified.py — Fine-tune YOLO pada gambar MOWA-rectified (kondisi "rectify-both").

Latar: evaluasi A/B menunjukkan MOWA-rectified LEBIH BURUK saat model diuji apa adanya,
karena detektor dilatih pada gambar ASLI (terdistorsi) lalu diuji pada gambar rectified —
domain mismatch train/test. Deep-research (KITTI-360 fisheye benchmark; FisheyeYOLO/WoodScape)
menyimpulkan rektifikasi hanya membantu bila detektor DI-FINE-TUNE pada domain rectified.

Script ini fine-tune bobot terbaik (default YOLOv8m PIO) pada:
  train: data/rectified/pio_train/images   (+ label warp)
  val  : data/rectified/pio_val/images     (+ label warp)

Lalu bobot hasil dievaluasi ulang oleh src/eval_detection.py (kondisi B').

Pemakaian:
  # (1) Fine-tune rectify-both PIO (perilaku default, build_yaml PIO-rectified):
  .venv-yolo/Scripts/python.exe src/finetune_rectified.py \
      --weights "train model/runs_compare/cmp_yolov8m/weights/best.pt" \
      --epochs 40

  # (2) Retrain augmentasi radial (pakai data.yaml eksternal apa adanya):
  .venv-yolo/Scripts/python.exe src/finetune_rectified.py \
      --weights "train model/runs_compare/cmp_yolov8m/weights/best.pt" \
      --data data/augmented/pio_train_radial/_radial_pio.yaml \
      --project "train model/runs_radial" --name ft_radial_yolov8m --epochs 40
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def build_yaml(out_path: Path, val_dir: Path | None = None) -> Path:
    # val_dir: override opsional direktori gambar val (default pio_val/images).
    val_path = val_dir if val_dir is not None else ROOT / "data" / "rectified" / "pio_val" / "images"
    cfg = {
        "path": str(ROOT),
        "train": [str(ROOT / "data" / "rectified" / "pio_train" / "images")],
        "val": [str(val_path)],
        "nc": 1,
        "names": {0: "pollo"},
    }
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Fine-tune YOLO pada gambar rectified.")
    ap.add_argument("--weights", type=Path,
                    default=ROOT / "train model" / "runs_compare" / "cmp_yolov8m" / "weights" / "best.pt")
    ap.add_argument("--epochs", type=int, default=20,
                    help="Cukup untuk adaptasi domain rectified (model sudah pretrained di PIO).")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=4, help="Kecil utk RTX 4060 8GB @960.")
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default="ft_rectified_yolov8m")
    ap.add_argument("--project", type=Path, default=ROOT / "train model" / "runs_rectified")
    ap.add_argument("--resume", action="store_true",
                    help="Lanjutkan dari last.pt run sebelumnya bila ada (tahan terhadap interupsi).")
    ap.add_argument("--data", type=Path, default=None,
                    help="Path data.yaml eksternal. Bila diisi, dipakai apa adanya dan "
                         "build_yaml PIO-rectified DILEWATI (mis. untuk retrain augmentasi radial).")
    ap.add_argument("--val", type=Path, default=None,
                    help="Override opsional direktori gambar val saat build_yaml default "
                         "(hanya dipakai bila --data TIDAK diberikan).")
    args = ap.parse_args()

    from ultralytics import YOLO

    if args.data is not None:
        data_yaml = args.data
        if not data_yaml.exists():
            raise SystemExit(f"[finetune] ERROR: data.yaml eksternal tidak ditemukan: {data_yaml}")
        print(f"[finetune] pakai data.yaml EKSTERNAL apa adanya -> {data_yaml}")
    else:
        data_yaml = build_yaml(ROOT / "data" / "rectified" / "_rectified_pio.yaml", val_dir=args.val)
    last_ckpt = args.project / args.name / "weights" / "last.pt"

    if args.resume and last_ckpt.exists():
        print(f"[finetune] RESUME dari {last_ckpt}")
        model = YOLO(str(last_ckpt))
        model.train(resume=True)
    else:
        print(f"[finetune] data.yaml -> {data_yaml}")
        print(f"[finetune] base weights -> {args.weights}")
        model = YOLO(str(args.weights))
        model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=str(args.project),
            name=args.name,
            exist_ok=True,
            patience=10,
            verbose=True,
        )
    best = args.project / args.name / "weights" / "best.pt"
    print(f"[finetune] SELESAI. best -> {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
