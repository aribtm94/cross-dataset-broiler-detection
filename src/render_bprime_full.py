"""
render_bprime_full.py — Gambar deteksi FULL-RESOLUTION kondisi B' (MOWA + fine-tune
"rectify-both") untuk SETIAP gambar di ketiga dataset.

Kondisi B' = bobot fine-tune-on-rectified (ft_rectified_yolov8m) dijalankan pada
gambar hasil rektifikasi MOWA. Tiap keluaran adalah gambar rectified ukuran ASLI
dengan kotak PREDIKSI (oranye) + GROUND-TRUTH warp (hijau) digambar di atasnya,
plus strip judul tipis (dataset, nama file, jumlah deteksi/GT).

Berbeda dari compare_condition_panels.py (yang membuat panel 3-kolom A/B/B' berukuran
kecil): script ini fokus HANYA B' dan menyimpan gambar penuh per berkas, cocok untuk
ditinjau detail atau dilampirkan ke laporan.

Meng-impor ulang helper dari compare_condition_panels.py (build_index, run_predict,
read_yolo, load_font) — tidak mengimplementasi ulang inferensi/pemetaan dataset.

Pakai .venv-yolo (butuh GPU untuk kecepatan; CPU juga jalan tapi lambat). Contoh:
  .venv-yolo/Scripts/python.exe src/render_bprime_full.py
  .venv-yolo/Scripts/python.exe src/render_bprime_full.py --datasets pio_val --limit 20
  .venv-yolo/Scripts/python.exe src/render_bprime_full.py --no-gt   # hanya prediksi

Keluaran:
  reports/bprime_full/<dataset>/<stem>.jpg
  reports/bprime_full/<dataset>/_manifest.json  (ringkasan jumlah, rata-rata deteksi)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

# Kurangi fragmentasi VRAM pada kartu 8GB (harus diset sebelum torch/cuda pertama kali).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from compare_condition_panels import (  # noqa: E402  (perlu sys.path dulu)
    build_index, run_predict, read_yolo, load_font,
    FINETUNE_WEIGHTS, GT_COLOR, PRED_COLOR, NAVY, MUTED, WHITE, PANEL_BG,
    IMAGE_EXTS,
)

OUT_ROOT = ROOT / "reports" / "bprime_full"

# strip judul tipis di atas tiap gambar
BAND_H = 46


def draw_full(rect_img_path: Path,
              preds: List[Tuple[float, float, float, float, float]],
              gts: List[Tuple[float, float, float, float]],
              dataset_display: str, stem: str, draw_gt: bool,
              f_hdr, f_sub) -> Tuple[Image.Image, int, int]:
    """Gambar rectified ukuran asli + kotak; strip judul di atas. Return (img, n_pred, n_gt)."""
    im = Image.open(rect_img_path).convert("RGB")
    W0, H0 = im.size
    d = ImageDraw.Draw(im, "RGBA")

    # ground-truth (hijau) lebih dulu agar prediksi di atasnya
    n_gt = 0
    if draw_gt:
        for cx, cy, w, h in gts:
            x1 = (cx - w / 2) * W0
            y1 = (cy - h / 2) * H0
            x2 = (cx + w / 2) * W0
            y2 = (cy + h / 2) * H0
            d.rectangle([x1, y1, x2, y2], outline=GT_COLOR + (255,), width=2)
        n_gt = len(gts)

    # prediksi (oranye) — sudah dalam koordinat piksel gambar rectified
    for x1, y1, x2, y2, _c in preds:
        d.rectangle([x1, y1, x2, y2], outline=PRED_COLOR + (255,), width=3)
    n_pred = len(preds)

    # tempel di kanvas dengan strip judul di atas
    canvas = Image.new("RGB", (W0, H0 + BAND_H), WHITE)
    canvas.paste(im, (0, BAND_H))
    dd = ImageDraw.Draw(canvas)
    dd.rectangle([0, 0, W0, BAND_H], fill=PANEL_BG)
    dd.rectangle([0, BAND_H - 2, W0, BAND_H], fill=(217, 226, 234))
    dd.text((14, 6), f"B' fine-tune (rectify-both) — {dataset_display}", fill=NAVY, font=f_hdr)
    info = f"{stem}   deteksi={n_pred}" + (f"   GT={n_gt}" if draw_gt else "")
    # info rata kanan
    tw = dd.textlength(info, font=f_sub)
    dd.text((max(14, W0 - int(tw) - 14), 24), info, fill=MUTED, font=f_sub)
    return canvas, n_pred, n_gt


def main() -> int:
    ap = argparse.ArgumentParser(description="Render gambar deteksi B' full-resolution untuk semua dataset.")
    ap.add_argument("--datasets", default="", help="Subset id dipisah koma (default: semua).")
    ap.add_argument("--limit", type=int, default=0, help="Maks gambar per dataset (0 = semua).")
    ap.add_argument("--weights", type=Path, default=FINETUNE_WEIGHTS, help="Bobot B' (default ft_rectified_yolov8m).")
    ap.add_argument("--device", default="0", help="'0' untuk GPU cuda:0, 'cpu' untuk CPU.")
    ap.add_argument("--no-gt", action="store_true", help="Jangan gambar kotak ground-truth (hanya prediksi).")
    ap.add_argument("--ext", default=".jpg", help="Ekstensi gambar keluaran (default .jpg).")
    args = ap.parse_args()

    if not args.weights.exists():
        print(f"ERROR: bobot B' tidak ditemukan: {args.weights}", file=sys.stderr)
        return 2

    from ultralytics import YOLO

    f_hdr = load_font(22, bold=True)
    f_sub = load_font(16, bold=False)

    index = build_index()
    want = [d.strip() for d in args.datasets.split(",") if d.strip()]
    targets = {k: v for k, v in index.items() if not want or k in want}
    if not targets:
        print(f"ERROR: tidak ada dataset cocok dengan --datasets={args.datasets}", file=sys.stderr)
        return 2

    print(f"[bprime] bobot B' = {args.weights}")
    model = YOLO(str(args.weights))

    grand = {}
    for ds_id, ds in targets.items():
        entries = ds["entries"]
        if args.limit > 0:
            entries = entries[: args.limit]
        # hanya proses yang gambar rectified-nya ada
        entries = [e for e in entries if e["rect_img"] is not None and Path(e["rect_img"]).exists()]
        if not entries:
            print(f"[bprime] {ds_id}: tidak ada gambar rectified, dilewati.")
            continue

        out_dir = OUT_ROOT / ds_id
        out_dir.mkdir(parents=True, exist_ok=True)
        rect_paths = [Path(e["rect_img"]) for e in entries]

        print(f"[bprime] {ds_id}: prediksi {len(rect_paths)} gambar rectified ...")
        t0 = time.time()
        preds_all = run_predict(model, rect_paths, args.device)

        n_ok = 0
        total_pred = 0
        for e, preds in zip(entries, preds_all):
            gts = read_yolo(Path(e["rect_lbl"])) if (not args.no_gt and e.get("rect_lbl")) else []
            try:
                img, n_pred, n_gt = draw_full(
                    Path(e["rect_img"]), preds, gts, ds["display"], e["stem"],
                    draw_gt=not args.no_gt, f_hdr=f_hdr, f_sub=f_sub,
                )
            except Exception as ex:  # noqa: BLE001
                print(f"  GAGAL {e['stem']}: {ex}", file=sys.stderr)
                continue
            img.save(out_dir / (e["stem"] + args.ext), quality=90)
            n_ok += 1
            total_pred += n_pred
            if n_ok % 50 == 0 or n_ok == len(entries):
                dt = time.time() - t0
                print(f"  [{n_ok}/{len(entries)}] ({dt:.1f}s, {dt / max(1, n_ok):.3f}s/img)")

        manifest = {
            "dataset": ds_id,
            "display": ds["display"],
            "condition": "B' (MOWA rectified + fine-tune rectify-both)",
            "weights": str(args.weights),
            "images_written": n_ok,
            "total_predictions": total_pred,
            "mean_predictions_per_image": round(total_pred / max(1, n_ok), 2),
            "draw_gt": not args.no_gt,
            "seconds": round(time.time() - t0, 2),
        }
        (out_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        grand[ds_id] = manifest
        print(f"[bprime] {ds_id} SELESAI: {n_ok} gambar -> {out_dir}")

    (OUT_ROOT).mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "_summary.json").write_text(json.dumps(grand, indent=2), encoding="utf-8")
    print(f"[bprime] semua selesai -> {OUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
