"""
anomaly_example_overlay.py — Contoh penerapan deteksi anomali (voting ensemble ala paper cattle).

Menggambar SATU gambar PIO padat dengan:
  - semua bbox broiler normal (abu-abu tipis),
  - bbox ANOMALI disorot: warning (amber) & critical (merah), tebal,
  - label berat estimasi (g) pada tiap anomali,
  - inset ZOOM pada satu anomali critical (crop diperbesar) untuk lihat detail,
  - kotak info: metode, median konteks, jumlah anomali per level.

Anomali dari kolom `ensemble_level`/`vote_count` di features/weight_estimates_ensemble.csv
(lihat src/anomaly_ensemble.py). Skor = |log(berat / median-konteks)| → menandai ayam yang
ukuran/berat-relatifnya menyimpang dalam konteks gambar padat.

Pakai .venv-yolo (Pillow). Contoh:
  .venv-yolo/Scripts/python.exe src/anomaly_example_overlay.py
  .venv-yolo/Scripts/python.exe src/anomaly_example_overlay.py --image C-W3-V0023.jpg
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image, ImageDraw

from compare_condition_panels import load_font  # reuse font loader

csv.field_size_limit(10 ** 7)

ROOT = Path(__file__).resolve().parents[1]
ENSEMBLE_CSV = ROOT / "features" / "weight_estimates_ensemble.csv"
VAL_IMG_DIR = ROOT / "data" / "images" / "val"
OUT_DIR = ROOT / "reports" / "anomaly_example"
WEIGHT_KEY = "radial_depth_median_estimated_weight_g"

NAVY = (15, 42, 67)
INK = (30, 41, 59)
MUTED = (90, 104, 122)
WHITE = (255, 255, 255)
NORMAL = (150, 158, 168)       # abu-abu — bbox normal
WARN = (233, 162, 59)          # amber — warning (2 voter)
CRIT = (208, 58, 46)           # merah — critical (>=3 voter)
INSET_LINE = (0, 200, 255)     # cyan — penanda sumber inset

MARGIN = 26
HEADER_H = 92
INFO_W = 360                   # panel info kanan
INSET_W = 330                  # lebar inset zoom
MAX_IMG_W = 1500               # lebar tampilan gambar utama


def to_f(v, d=math.nan) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else d
    except (TypeError, ValueError):
        return d


def load_rows(image_name: str) -> List[Dict[str, str]]:
    header = ENSEMBLE_CSV.open(encoding="utf-8").readline().strip().split(",")
    idx = {k: i for i, k in enumerate(header)}
    rows = []
    with ENSEMBLE_CSV.open(encoding="utf-8") as fh:
        r = csv.reader(fh)
        next(r)
        for row in r:
            if row[idx["image"]] == image_name:
                rows.append({k: row[idx[k]] for k in idx})
    return rows


def pick_default_image() -> str:
    """Cari gambar val padat dengan banyak anomali critical (jika --image tak diberi)."""
    header = ENSEMBLE_CSV.open(encoding="utf-8").readline().strip().split(",")
    idx = {k: i for i, k in enumerate(header)}
    val_stems = {p.stem for p in VAL_IMG_DIR.iterdir()} if VAL_IMG_DIR.is_dir() else set()
    from collections import Counter
    tot, crit = Counter(), Counter()
    with ENSEMBLE_CSV.open(encoding="utf-8") as fh:
        r = csv.reader(fh)
        next(r)
        for row in r:
            img = row[idx["image"]]
            if Path(img).stem not in val_stems:
                continue
            tot[img] += 1
            if row[idx["ensemble_level"]] == "critical":
                crit[img] += 1
    cand = [(im, crit[im], tot[im]) for im in tot if tot[im] >= 200 and crit[im] >= 8]
    cand.sort(key=lambda t: (t[1], t[2]), reverse=True)
    return cand[0][0] if cand else "C-W3-V0016.jpg"


def draw_label(d: ImageDraw.ImageDraw, x: float, y: float, text: str, fg, bg, font):
    tb = d.textbbox((0, 0), text, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    pad = 3
    yy = max(0, y - th - 2 * pad)
    d.rectangle([x, yy, x + tw + 2 * pad, yy + th + 2 * pad], fill=bg)
    d.text((x + pad, yy + pad), text, fill=fg, font=font)


def main() -> int:
    ap = argparse.ArgumentParser(description="Contoh overlay deteksi anomali (ensemble).")
    ap.add_argument("--image", default="", help="Nama file gambar val (mis. C-W3-V0016.jpg).")
    args = ap.parse_args()

    if not ENSEMBLE_CSV.exists():
        print(f"ERROR: {ENSEMBLE_CSV} tidak ada. Jalankan src/anomaly_ensemble.py dulu.")
        return 2

    image_name = args.image or pick_default_image()
    img_path = VAL_IMG_DIR / image_name
    if not img_path.exists():
        print(f"ERROR: gambar tidak ditemukan: {img_path}")
        return 2

    rows = load_rows(image_name)
    if not rows:
        print(f"ERROR: tak ada baris fitur untuk {image_name}")
        return 2

    weights = [to_f(r.get(WEIGHT_KEY)) for r in rows if to_f(r.get(WEIGHT_KEY)) > 0]
    ctx_median = statistics.median(weights) if weights else 0.0

    anomalies = [r for r in rows if r.get("ensemble_level") in ("warning", "critical")]
    n_warn = sum(1 for r in anomalies if r["ensemble_level"] == "warning")
    n_crit = sum(1 for r in anomalies if r["ensemble_level"] == "critical")

    # ---- muat & skala gambar utama ----
    im = Image.open(img_path).convert("RGB")
    W0, H0 = im.size
    scale = min(1.0, MAX_IMG_W / W0)
    disp_w, disp_h = int(W0 * scale), int(H0 * scale)
    main = im.resize((disp_w, disp_h), Image.BILINEAR)
    d = ImageDraw.Draw(main, "RGBA")

    f_lbl = load_font(13, bold=True)

    # bbox normal dulu (tipis)
    for r in rows:
        if r.get("ensemble_level") in ("warning", "critical"):
            continue
        x1, y1, x2, y2 = (to_f(r["x1"]) * scale, to_f(r["y1"]) * scale,
                          to_f(r["x2"]) * scale, to_f(r["y2"]) * scale)
        d.rectangle([x1, y1, x2, y2], outline=NORMAL + (170,), width=1)

    # pilih anomali untuk inset: prioritaskan yang INTERIOR (bukan sliver tepi) & box wajar,
    # karena anomali di tepi frame sering artefak (ayam terpotong -> berat kecil palsu).
    def box_area(r):
        return (to_f(r["x2"]) - to_f(r["x1"])) * (to_f(r["y2"]) - to_f(r["y1"]))

    def not_edge(r, m=25):
        return (to_f(r["x1"]) > m and to_f(r["y1"]) > m
                and to_f(r["x2"]) < W0 - m and to_f(r["y2"]) < H0 - m)

    inset_src = None
    interior = [r for r in anomalies if not_edge(r) and box_area(r) >= 1500]
    if interior:
        # utamakan critical, lalu box terbesar (ayam yang benar-benar beda ukuran)
        interior.sort(key=lambda r: (r["ensemble_level"] == "critical", box_area(r)), reverse=True)
        inset_src = interior[0]
    elif anomalies:
        inset_src = sorted(anomalies, key=box_area, reverse=True)[0]

    # gambar anomali (tebal + label berat)
    for r in anomalies:
        lvl = r["ensemble_level"]
        col = CRIT if lvl == "critical" else WARN
        x1, y1, x2, y2 = (to_f(r["x1"]) * scale, to_f(r["y1"]) * scale,
                          to_f(r["x2"]) * scale, to_f(r["y2"]) * scale)
        d.rectangle([x1, y1, x2, y2], outline=col + (255,), width=3)
        w_g = to_f(r.get(WEIGHT_KEY))
        arrow = "▲" if w_g > ctx_median else "▼"
        draw_label(d, x1, y1, f"{w_g:.0f}g {arrow}", WHITE, col, f_lbl)

    # tandai kotak sumber inset
    if inset_src is not None:
        ix1, iy1, ix2, iy2 = (to_f(inset_src["x1"]) * scale, to_f(inset_src["y1"]) * scale,
                              to_f(inset_src["x2"]) * scale, to_f(inset_src["y2"]) * scale)
        d.rectangle([ix1 - 3, iy1 - 3, ix2 + 3, iy2 + 3], outline=INSET_LINE + (255,), width=2)

    # ---- inset zoom ----
    inset_img = None
    if inset_src is not None:
        bx1, by1, bx2, by2 = (to_f(inset_src["x1"]), to_f(inset_src["y1"]),
                              to_f(inset_src["x2"]), to_f(inset_src["y2"]))
        bw, bh = bx2 - bx1, by2 - by1
        pad = max(bw, bh) * 1.6 + 20
        cx1 = max(0, int(bx1 - pad)); cy1 = max(0, int(by1 - pad))
        cx2 = min(W0, int(bx2 + pad)); cy2 = min(H0, int(by2 + pad))
        crop = im.crop((cx1, cy1, cx2, cy2))
        cw, ch = crop.size
        s2 = INSET_W / cw
        inset_img = crop.resize((INSET_W, max(1, int(ch * s2))), Image.BILINEAR)
        di = ImageDraw.Draw(inset_img, "RGBA")
        # gambar ulang kotak anomali di dalam crop
        rx1 = (bx1 - cx1) * s2; ry1 = (by1 - cy1) * s2
        rx2 = (bx2 - cx1) * s2; ry2 = (by2 - cy1) * s2
        col = CRIT if inset_src["ensemble_level"] == "critical" else WARN
        di.rectangle([rx1, ry1, rx2, ry2], outline=col + (255,), width=3)

    # ---- komposisi kanvas ----
    right_w = max(INFO_W, INSET_W)
    total_w = MARGIN * 3 + disp_w + right_w
    # tinggi kolom kanan (inset + caption + kotak ringkasan) supaya legenda tak terpotong
    box_h = 300
    right_col_h = box_h
    if inset_img is not None:
        right_col_h = 26 + inset_img.height + 62 + box_h
    total_h = HEADER_H + max(disp_h, right_col_h) + MARGIN
    canvas = Image.new("RGB", (total_w, total_h), WHITE)
    cd = ImageDraw.Draw(canvas)

    f_hdr = load_font(27, bold=True)
    f_sub = load_font(14, bold=False)
    f_h2 = load_font(16, bold=True)
    f_body = load_font(13, bold=False)
    f_legb = load_font(13, bold=True)

    # header
    cd.rectangle([0, 0, total_w, HEADER_H], fill=NAVY)
    cd.text((MARGIN, 16), "Contoh Deteksi Anomali Broiler — Voting Ensemble (adaptasi paper cattle)",
            fill=WHITE, font=f_hdr)
    cd.text((MARGIN, 54),
            f"PIO val · {image_name} · konteks per-gambar (bbox padat) · median konteks = {ctx_median:.0f} g",
            fill=(200, 214, 226), font=f_sub)

    # gambar utama
    canvas.paste(main, (MARGIN, HEADER_H))
    cd.rectangle([MARGIN, HEADER_H, MARGIN + disp_w, HEADER_H + disp_h],
                 outline=(210, 218, 226), width=1)

    # panel kanan
    rx = MARGIN * 2 + disp_w
    ry = HEADER_H
    # inset dulu
    if inset_img is not None:
        inset_lvl = inset_src["ensemble_level"]
        cd.rectangle([rx, ry, rx + INSET_W, ry + 26], fill=INSET_LINE)
        cd.text((rx + 8, ry + 4), f"ZOOM anomali ({inset_lvl})", fill=NAVY, font=f_legb)
        canvas.paste(inset_img, (rx, ry + 26))
        cd.rectangle([rx, ry + 26, rx + inset_img.width, ry + 26 + inset_img.height],
                     outline=(210, 218, 226), width=1)
        w_g = to_f(inset_src.get(WEIGHT_KEY))
        rel = (w_g / ctx_median - 1) * 100 if ctx_median else 0
        cd.text((rx, ry + 34 + inset_img.height),
                f"berat est. {w_g:.0f} g  ({rel:+.0f}% vs median)  ·  votes={inset_src.get('vote_count')}",
                fill=INK, font=f_body)
        ry = ry + 26 + inset_img.height + 62

    # kotak ringkasan
    cd.rectangle([rx, ry, rx + INFO_W, ry + box_h], fill=(245, 248, 250), outline=(210, 218, 226), width=1)
    cd.rectangle([rx, ry, rx + INFO_W, ry + 32], fill=NAVY)
    cd.text((rx + 10, ry + 7), "Ringkasan", fill=WHITE, font=f_h2)
    lines = [
        (f"Total broiler terdeteksi: {len(rows)}", INK),
        (f"Anomali (≥2 voter): {len(anomalies)}", INK),
        (f"  • warning (2 voter): {n_warn}", WARN),
        (f"  • critical (≥3 voter): {n_crit}", CRIT),
        ("", INK),
        ("Voter: z-score · IQR · MAD robust · autoencoder", MUTED),
        ("Skor = |log(berat / median konteks)|", MUTED),
        ("Anomali = ukuran menyimpang di gambar padat,", MUTED),
        ("bukan diagnosa medis.", MUTED),
    ]
    yy = ry + 42
    for txt, col in lines:
        cd.text((rx + 12, yy), txt, fill=col, font=f_body)
        yy += 22
    # legenda warna
    ly = yy + 6
    for col, name in [(NORMAL, "normal"), (WARN, "warning"), (CRIT, "critical")]:
        cd.rectangle([rx + 12, ly + 2, rx + 30, ly + 14], outline=col, width=3)
        cd.text((rx + 38, ly), name, fill=INK, font=f_legb)
        ly += 20

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{Path(image_name).stem}_anomaly.png"
    canvas.save(out_path)
    print(f"[anomaly] {image_name}: total={len(rows)} anom={len(anomalies)} "
          f"(warn={n_warn} crit={n_crit}) -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
