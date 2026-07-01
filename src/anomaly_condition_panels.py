"""
anomaly_condition_panels.py — Perbandingan DETEKSI ANOMALI lintas kondisi A / B / B'.

Untuk beberapa gambar dari 3 dataset, membuat panel 3-kolom yang menjalankan voting
ensemble anomali (ala paper cattle-outlier) pada KOTAK PREDIKSI tiap kondisi:
  A  — baseline  @ gambar asli
  B  — baseline  @ gambar MOWA-rectified
  B' — fine-tune @ gambar MOWA-rectified

Karena anomali dihitung dari kotak PREDIKSI (bukan GT), B dan B' BERBEDA: bobot detektor
berbeda -> kotak berbeda -> anomali berbeda. Inilah yang membuat perbandingan bermakna.

Metrik anomali = UKURAN-RELATIF (bukan gram), disamakan lintas dataset supaya PIO & eksternal
sebanding: skor = |log(size / median-konteks)| dengan size = sqrt(area bbox). Konteks = gambar
itu sendiri. Empat voter (z-score, IQR fence, MAD robust, percentile P97), ditandai bila
>=2 voter setuju (critical bila >=3) — sama semangat dengan src/anomaly_ensemble.py.

Untuk PIO (punya week) label anomali juga menyertakan estimasi gram kasar = Cobb500-week x rasio
ukuran; untuk eksternal hanya rasio ukuran (tanpa klaim gram).

Pakai .venv-yolo. Contoh:
  .venv-yolo/Scripts/python.exe src/anomaly_condition_panels.py
  .venv-yolo/Scripts/python.exe src/anomaly_condition_panels.py --per-dataset 2
"""
from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

from compare_condition_panels import (
    BASELINE_WEIGHTS, FINETUNE_WEIGHTS, build_index, load_font, run_predict,
)
from common import cobb_weight_for_age, parse_filename_metadata

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "anomaly_condition_panels"

# palet selaras contoh anomali tunggal
NAVY = (15, 42, 67)
INK = (30, 41, 59)
MUTED = (90, 104, 122)
WHITE = (255, 255, 255)
NORMAL = (150, 158, 168)
WARN = (233, 162, 59)
CRIT = (208, 58, 46)
ACCENT = {"A": (42, 157, 143), "B": (233, 162, 59), "B'": (28, 114, 147)}

CELL_W = 640
GAP = 16
MARGIN = 22
HEADER_H = 96
CAPTION_H = 60

# parameter voter (mengikuti semangat anomaly_ensemble.py; sedikit dilonggarkan agar
# voter beririsan pada outlier sejati — kotak PREDIKSI lebih seragam drpd GT+berat).
MIN_CTX = 12          # butuh >= ini kotak untuk membentuk konteks statistik
Z_THRESH = 2.0
ROBUST_THRESH = 3.0
AE_TAIL_P = 0.95
VOTE_MAJORITY = 2

# pilihan gambar default (sisanya diisi otomatis dari yang terpadat)
DEFAULT_PICKS = {
    "pio_val": ["C-W3-V0007"],
    "broiler_instance_seg": [],
    "chicken_detection_fum": [],
}


def _median(v):
    return statistics.median(v) if v else 0.0


def _percentile(vals: List[float], p: float) -> float:
    s = sorted(vals)
    if not s:
        return math.inf
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] * (hi - k) + s[hi] * (k - lo)


def ensemble_flags(boxes: List[Tuple[float, float, float, float, float]]
                   ) -> Tuple[List[str], float]:
    """Kembalikan (levels, median_area). level per box: normal/warning/critical.

    Metrik = AREA bbox (analog ellipse_area di pipeline asli; sebaran log lebih lebar drpd
    sqrt-area). Voting: z-score, IQR, MAD robust, percentile pada |log(area/med)|.
    """
    n = len(boxes)
    levels = ["normal"] * n
    sizes = [max(1.0, (x2 - x1) * (y2 - y1)) for (x1, y1, x2, y2, _c) in boxes]
    if n < MIN_CTX:
        return levels, (_median(sizes) if sizes else 0.0)

    mean_v = sum(sizes) / n
    var = sum((v - mean_v) ** 2 for v in sizes) / (n - 1)
    std = math.sqrt(var)
    med = _median(sizes)
    q1, q3 = _percentile(sizes, 0.25), _percentile(sizes, 0.75)
    iqr = q3 - q1
    low_f, high_f = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mad = _median([abs(v - med) for v in sizes]) or 0.0
    scores = [abs(math.log(s / med)) if (med > 0 and s > 0) else 0.0 for s in sizes]
    p_thr = _percentile(scores, AE_TAIL_P)

    for i, s in enumerate(sizes):
        votes = 0
        if std > 0 and abs((s - mean_v) / std) >= Z_THRESH:
            votes += 1
        if s < low_f or s > high_f:
            votes += 1
        if mad > 0 and abs(0.6745 * (s - med) / mad) >= ROBUST_THRESH:
            votes += 1
        if scores[i] >= p_thr and scores[i] > 0:
            votes += 1
        if votes >= 3:
            levels[i] = "critical"
        elif votes >= VOTE_MAJORITY:
            levels[i] = "warning"
    return levels, med


def draw_label(d, x, y, text, fg, bg, font):
    tb = d.textbbox((0, 0), text, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    pad = 3
    yy = y - th - 2 * pad
    if yy < 0:
        yy = y + 2  # kalau mepet atas, taruh di dalam kotak
    d.rectangle([x, yy, x + tw + 2 * pad, yy + th + 2 * pad], fill=bg)
    d.text((x + pad, yy + pad), text, fill=fg, font=font)


def render_condition(img_path: Path,
                     boxes: List[Tuple[float, float, float, float, float]],
                     cobb_g: Optional[float]) -> Tuple[Image.Image, int, int, int]:
    """Render satu kolom kondisi: kotak normal abu-abu + anomali disorot + label rasio."""
    im = Image.open(img_path).convert("RGB")
    W0, H0 = im.size
    scale = CELL_W / W0
    cell_h = max(1, int(round(H0 * scale)))
    im = im.resize((CELL_W, cell_h), Image.BILINEAR)
    d = ImageDraw.Draw(im, "RGBA")
    f_lbl = load_font(13, bold=True)

    levels, med_size = ensemble_flags(boxes)

    # normal dulu (tipis)
    for (x1, y1, x2, y2, _c), lvl in zip(boxes, levels):
        if lvl != "normal":
            continue
        d.rectangle([x1 * scale, y1 * scale, x2 * scale, y2 * scale],
                    outline=NORMAL + (160,), width=1)

    n_warn = n_crit = 0
    for (x1, y1, x2, y2, _c), lvl in zip(boxes, levels):
        if lvl == "normal":
            continue
        col = CRIT if lvl == "critical" else WARN
        if lvl == "critical":
            n_crit += 1
        else:
            n_warn += 1
        d.rectangle([x1 * scale, y1 * scale, x2 * scale, y2 * scale],
                    outline=col + (255,), width=3)
        area = max(1.0, (x2 - x1) * (y2 - y1))
        # rasio LINEAR (ukuran tubuh) = akar dari rasio area
        lin = math.sqrt(area / med_size) if med_size > 0 else 1.0
        rel = (area / med_size - 1) * 100 if med_size > 0 else 0.0  # rasio area utk %
        arrow = "▲" if area >= med_size else "▼"
        if cobb_g:  # PIO: estimasi gram kasar = cobb-week x rasio ukuran linear
            g = cobb_g * lin
            txt = f"{g:.0f}g {arrow}"
        else:
            txt = f"{rel:+.0f}% {arrow}"
        draw_label(d, x1 * scale, y1 * scale, txt, WHITE, col, f_lbl)

    return im, len(boxes), n_warn, n_crit


def compose(dataset_display: str, stem: str, cobb_g: Optional[float],
            cols: List[Tuple[Image.Image, int, int, int]]) -> Image.Image:
    labels = ["A", "B", "B'"]
    sub = ["baseline · asli", "baseline · rectified", "fine-tune · rectified"]
    cell_h = max(c[0].height for c in cols)
    total_w = MARGIN * 2 + CELL_W * 3 + GAP * 2
    total_h = HEADER_H + CAPTION_H + cell_h + MARGIN
    canvas = Image.new("RGB", (total_w, total_h), WHITE)
    d = ImageDraw.Draw(canvas)

    f_hdr = load_font(26, bold=True)
    f_sub = load_font(14, bold=False)
    f_cap = load_font(19, bold=True)
    f_capS = load_font(13, bold=False)
    f_leg = load_font(13, bold=True)

    d.rectangle([0, 0, total_w, HEADER_H], fill=NAVY)
    d.text((MARGIN, 14), f"Perbandingan Deteksi Anomali A/B/B' — {dataset_display}",
           fill=WHITE, font=f_hdr)
    unit = "estimasi gram (Cobb500)" if cobb_g else "ukuran-relatif (tanpa gram)"
    d.text((MARGIN, 50),
           f"{stem} · anomali dari kotak PREDIKSI tiap kondisi · voting ensemble · label: {unit}",
           fill=(200, 214, 226), font=f_sub)
    # legend kanan
    lx = total_w - MARGIN - 430
    d.rectangle([lx, 20, lx + 16, 36], outline=NORMAL, width=3)
    d.text((lx + 22, 18), "normal", fill=WHITE, font=f_leg)
    d.rectangle([lx + 110, 20, lx + 126, 36], outline=WARN, width=3)
    d.text((lx + 132, 18), "warning", fill=WHITE, font=f_leg)
    d.rectangle([lx + 235, 20, lx + 251, 36], outline=CRIT, width=3)
    d.text((lx + 257, 18), "critical", fill=WHITE, font=f_leg)

    x = MARGIN
    cap_y = HEADER_H
    for (cell, n_tot, n_warn, n_crit), lab, sb in zip(cols, labels, sub):
        ac = ACCENT[lab]
        d.rectangle([x, cap_y, x + CELL_W, cap_y + CAPTION_H], fill=ac)
        htext = NAVY if lab == "B" else WHITE  # amber -> teks navy
        d.text((x + 12, cap_y + 6), f"{lab} — {sb}", fill=htext, font=f_cap)
        anom = n_warn + n_crit
        note = "" if n_tot >= MIN_CTX else "  (konteks < min)"
        d.text((x + 12, cap_y + 34),
               f"deteksi: {n_tot}   anomali: {anom} (w{n_warn}/c{n_crit}){note}",
               fill=htext, font=f_capS)
        canvas.paste(cell, (x, cap_y + CAPTION_H))
        d.rectangle([x, cap_y + CAPTION_H, x + CELL_W, cap_y + CAPTION_H + cell.height],
                    outline=(210, 218, 226), width=1)
        x += CELL_W + GAP
    return canvas


def pick_entries(ds_id: str, ds: Dict, per_dataset: int) -> List[Dict]:
    """Pilih entri: default dulu, lalu isi dari yang GT-nya terpadat (proxy kepadatan)."""
    entries = ds["entries"]
    by_stem = {e["stem"]: e for e in entries}
    chosen: List[Dict] = []
    for want in DEFAULT_PICKS.get(ds_id, []):
        if want in by_stem:
            chosen.append(by_stem[want])
    # sisanya: urut berdasar jumlah baris label rectified (kepadatan)
    def gt_count(e):
        lbl = e["rect_lbl"]
        if not lbl.exists():
            return 0
        return sum(1 for ln in lbl.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip())
    rest = sorted((e for e in entries if e not in chosen and e["orig_img"] and e["rect_img"].exists()),
                  key=gt_count, reverse=True)
    for e in rest:
        if len(chosen) >= per_dataset:
            break
        chosen.append(e)
    return chosen[:per_dataset]


def main() -> int:
    ap = argparse.ArgumentParser(description="Panel perbandingan anomali A/B/B'.")
    ap.add_argument("--datasets", nargs="*",
                    default=["pio_val", "broiler_instance_seg", "chicken_detection_fum"])
    ap.add_argument("--per-dataset", type=int, default=2, help="Jumlah gambar per dataset.")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    for w in (BASELINE_WEIGHTS, FINETUNE_WEIGHTS):
        if not w.exists():
            print(f"ERROR: bobot tidak ditemukan: {w}")
            return 2

    from ultralytics import YOLO
    print("[load] baseline:", BASELINE_WEIGHTS.name)
    base_model = YOLO(str(BASELINE_WEIGHTS))
    print("[load] finetune:", FINETUNE_WEIGHTS.name)
    ft_model = YOLO(str(FINETUNE_WEIGHTS))

    index = build_index()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for ds_id in args.datasets:
        if ds_id not in index:
            print(f"[skip] dataset tak dikenal: {ds_id}")
            continue
        ds = index[ds_id]
        chosen = pick_entries(ds_id, ds, args.per_dataset)
        print(f"\n[{ds_id}] {len(chosen)} gambar")

        orig = [e["orig_img"] for e in chosen]
        rect = [e["rect_img"] for e in chosen]
        preds_A = run_predict(base_model, orig, args.device)
        preds_B = run_predict(base_model, rect, args.device)
        preds_Bp = run_predict(ft_model, rect, args.device)

        for i, e in enumerate(chosen):
            meta = parse_filename_metadata(e["name"])
            cobb = None
            if meta.get("age_days") is not None:
                cobb = cobb_weight_for_age(meta["age_days"])
            cA = render_condition(e["orig_img"], preds_A[i], cobb)
            cB = render_condition(e["rect_img"], preds_B[i], cobb)
            cBp = render_condition(e["rect_img"], preds_Bp[i], cobb)
            panel = compose(ds["display"], e["stem"], cobb, [cA, cB, cBp])
            out_path = OUT_DIR / f"{ds_id}__{e['stem']}_anomaly_ABB.png"
            panel.save(out_path)
            print(f"  {e['stem']}: A(a={cA[2]+cA[3]}) B(a={cB[2]+cB[3]}) "
                  f"B'(a={cBp[2]+cBp[3]}) -> {out_path.name}")

    print(f"\n[selesai] output di {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
