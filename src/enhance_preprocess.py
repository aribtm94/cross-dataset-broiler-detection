"""
enhance_preprocess.py — Preprocessor fotometrik (CLAHE + unsharp mask) untuk dataset YOLO.

Tujuan (skripsi): melawan blur interpolasi hasil resampling MOWA sekaligus menaikkan
kontras objek kecil yang padat (ayam broiler saling berdempet). Rectification MOWA
memakai remap/grid_sample yang menghaluskan tepi (softening); tahap ini
mengembalikan ketajaman dan memperjelas batas antar-objek TANPA mengubah geometri —
murni fotometrik, sehingga bounding box lama tetap valid.

Dua operasi:
  - CLAHE (Contrast Limited Adaptive Histogram Equalization): dikerjakan HANYA pada
    kanal L (luminance) di ruang warna LAB, jadi warna & geometri tidak berubah.
    Kontras lokal naik -> objek kecil di area gelap/silau jadi lebih terpisah.
  - Unsharp mask: sharp = img*(1+amount) - blur(img)*amount. Mempertegas tepi yang
    dilembutkan oleh interpolasi.
  - mode=both (default): CLAHE dulu (perbaiki kontras), lalu unsharp (pertegas tepi).

Karena operasi ini geometry-preserving, label .txt YOLO DISALIN apa adanya (--labels).

Literatur: kombinasi CLAHE sebagai praproses di depan YOLO terbukti menaikkan mAP
deteksi objek kecil (mis. Kaur & Kaur 2023; berbagai studi deteksi small-object pada
citra kontras rendah menunjukkan CLAHE+sharpening memperbaiki recall objek kecil).

Contoh pemakaian (dari root proyek, pakai venv YOLO — TANPA GPU/torch):
  .venv-yolo/Scripts/python.exe src/enhance_preprocess.py \
      --input data/rectified/pio_val/images \
      --labels data/rectified/pio_val/labels \
      --output data/enhanced/pio_val \
      --mode both --clahe-clip 2.0 --clahe-grid 8 --unsharp-amount 1.0

Struktur output:
  <output>/images/*.jpg              citra hasil enhancement (resolusi & geometri sama)
  <output>/labels/*.txt              label YOLO (disalin apa adanya, jika --labels diberi)
  <output>/enhance_manifest.json     ringkasan run (mode, params, ok/failed, sec/img)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(input_dir: Path) -> List[Path]:
    """Daftar file gambar (terurut) di folder, disaring berdasar ekstensi."""
    return sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def apply_clahe(img_bgr: np.ndarray, clip: float, grid: int) -> np.ndarray:
    """CLAHE pada kanal L (LAB) saja -> warna & geometri terjaga.

    BGR -> LAB, equalize adaptif L, merge, kembali ke BGR.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def apply_unsharp(img_bgr: np.ndarray, amount: float, sigma: float) -> np.ndarray:
    """Unsharp mask: sharp = img*(1+amount) - blur*amount, di-clip ke uint8.

    sigma = radius Gaussian blur; amount = kekuatan penajaman.
    """
    blur = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharp = cv2.addWeighted(img_bgr, 1.0 + amount, blur, -amount, 0.0)
    return sharp  # addWeighted pada input uint8 sudah menghasilkan uint8 ter-saturasi


def enhance_image(img_bgr: np.ndarray, mode: str, clip: float, grid: int,
                  amount: float, sigma: float) -> np.ndarray:
    """Terapkan pipeline enhancement sesuai mode. Untuk both: CLAHE dulu, lalu unsharp."""
    out = img_bgr
    if mode in ("clahe", "both"):
        out = apply_clahe(out, clip, grid)
    if mode in ("unsharp", "both"):
        out = apply_unsharp(out, amount, sigma)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Preprocessor fotometrik CLAHE + unsharp mask untuk dataset YOLO "
                    "(geometry-preserving, tanpa GPU/torch).")
    ap.add_argument("--input", required=True, type=Path, help="Folder gambar sumber (images/).")
    ap.add_argument("--labels", type=Path, default=None,
                    help="Folder label YOLO (.txt) opsional; disalin apa adanya ke output.")
    ap.add_argument("--output", required=True, type=Path,
                    help="Folder output (dibuat: images/, labels/).")
    ap.add_argument("--mode", choices=["clahe", "unsharp", "both"], default="both",
                    help="Operasi yang diterapkan (default both: CLAHE lalu unsharp).")
    ap.add_argument("--clahe-clip", type=float, default=2.0, help="clipLimit CLAHE (default 2.0).")
    ap.add_argument("--clahe-grid", type=int, default=8,
                    help="Sisi tileGridSize CLAHE -> (grid, grid) (default 8).")
    ap.add_argument("--unsharp-amount", type=float, default=1.0,
                    help="Kekuatan unsharp mask (default 1.0).")
    ap.add_argument("--unsharp-sigma", type=float, default=1.0,
                    help="Radius Gaussian blur untuk unsharp mask (default 1.0).")
    ap.add_argument("--limit", type=int, default=0, help="Proses N gambar pertama saja (0 = semua).")
    ap.add_argument("--ext", default=".jpg", help="Ekstensi file output gambar (default .jpg).")
    args = ap.parse_args()

    if not args.input.is_dir():
        print(f"ERROR: --input bukan folder: {args.input}", file=sys.stderr)
        return 2
    if args.clahe_grid < 1:
        print(f"ERROR: --clahe-grid harus >= 1 (diberi {args.clahe_grid})", file=sys.stderr)
        return 2

    out_img_dir = args.output / "images"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    do_labels = args.labels is not None
    out_lbl_dir = args.output / "labels"
    if do_labels:
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(args.input)
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        print(f"ERROR: tidak ada gambar di {args.input}", file=sys.stderr)
        return 2

    ok, failed = 0, 0
    fail_names: List[str] = []
    t0 = time.time()
    for i, img_path in enumerate(images, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            failed += 1
            fail_names.append(img_path.name)
            print(f"  [{i}/{len(images)}] SKIP (tak terbaca): {img_path.name}", file=sys.stderr)
            continue

        try:
            out = enhance_image(img, args.mode, args.clahe_clip, args.clahe_grid,
                                args.unsharp_amount, args.unsharp_sigma)
        except Exception as e:  # noqa: BLE001 — laporkan, lanjut gambar berikutnya
            failed += 1
            fail_names.append(img_path.name)
            print(f"  [{i}/{len(images)}] GAGAL enhance {img_path.name}: {e}", file=sys.stderr)
            continue

        cv2.imwrite(str(out_img_dir / (img_path.stem + args.ext)), out)

        if do_labels:
            src_lbl = args.labels / (img_path.stem + ".txt")
            if src_lbl.exists():
                shutil.copy2(src_lbl, out_lbl_dir / (img_path.stem + ".txt"))

        ok += 1
        if i % 50 == 0 or i == len(images):
            dt = time.time() - t0
            print(f"  [{i}/{len(images)}] ok={ok} failed={failed} "
                  f"({dt:.1f}s, {dt / i:.3f}s/img)")

    total_dt = time.time() - t0
    manifest = {
        "input": str(args.input),
        "labels": str(args.labels) if args.labels else None,
        "output": str(args.output),
        "mode": args.mode,
        "params": {
            "clahe_clip": args.clahe_clip,
            "clahe_grid": args.clahe_grid,
            "unsharp_amount": args.unsharp_amount,
            "unsharp_sigma": args.unsharp_sigma,
        },
        "note": "Enhancement fotometrik (geometry-preserving): label disalin apa adanya, "
                "bbox tetap valid. CLAHE pada kanal-L LAB; unsharp mask menajamkan tepi "
                "yang dilembutkan resampling MOWA.",
        "total": len(images),
        "ok": ok,
        "failed": failed,
        "failed_names": fail_names,
        "seconds": round(total_dt, 2),
        "sec_per_img": round(total_dt / max(1, ok + failed), 4),
    }
    (args.output / "enhance_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[enhance_preprocess] SELESAI: ok={ok} failed={failed} -> {args.output}")
    print(f"[enhance_preprocess] manifest: {args.output / 'enhance_manifest.json'}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
