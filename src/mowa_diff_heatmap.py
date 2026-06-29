"""
mowa_diff_heatmap.py — Varian 3-kolom before/after MOWA + DIFFERENCE HEATMAP (1 per dataset).

Kolom:
  1. BEFORE  — gambar asli (terdistorsi)
  2. AFTER   — gambar hasil MOWA rectified
  3. DIFF    — heatmap magnitudo perpindahan piksel |after - before| (inferno) di atas
              backdrop rectified yang di-redup. Area terang = piksel paling banyak
              bergeser/berubah oleh warping (biasanya tepi & pinggir frame).

Tujuan: menyorot secara kuantitatif di mana MOWA paling banyak me-resample/menggeser,
memperkuat argumen skripsi soal blur-tepi & FOV-trim (resampling ganda) pada kondisi B/B'.

Tidak butuh GPU. Pakai .venv-yolo (Pillow + numpy). Gambar B dan B' identik (output MOWA sama).

Contoh:
  .venv-yolo/Scripts/python.exe src/mowa_diff_heatmap.py
  .venv-yolo/Scripts/python.exe src/mowa_diff_heatmap.py --pick pio_val=C-W2-V0003
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image, ImageDraw

from compare_condition_panels import build_index, load_font  # reuse
from mowa_before_after import DEFAULT_PICK  # reuse pilihan default

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "mowa_diff_heatmap"

NAVY = (15, 42, 67)
INK = (30, 41, 59)
MUTED = (100, 116, 139)
WHITE = (255, 255, 255)
PANEL_BG = (245, 248, 250)
BEFORE_AC = (192, 87, 70)
AFTER_AC = (42, 157, 143)
DIFF_AC = (28, 114, 147)

CELL_W = 620
GAP = 18
MARGIN = 24
HEADER_H = 84
CAPTION_H = 56

# titik kontrol colormap "inferno" (0..1) -> RGB, cukup untuk interpolasi mulus
_INFERNO = np.array([
    [0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99],
    [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164],
], dtype=np.float32)


def apply_inferno(norm: np.ndarray) -> np.ndarray:
    """norm: HxW dalam [0,1] -> HxWx3 uint8 via interpolasi titik kontrol inferno."""
    n = _INFERNO.shape[0] - 1
    pos = np.clip(norm, 0, 1) * n
    lo = np.floor(pos).astype(np.int32)
    hi = np.clip(lo + 1, 0, n)
    frac = (pos - lo)[..., None]
    out = _INFERNO[lo] * (1 - frac) + _INFERNO[hi] * frac
    return out.astype(np.uint8)


def load_rgb(path: Path, size: Optional[tuple] = None) -> Image.Image:
    im = Image.open(path).convert("RGB")
    if size is not None and im.size != size:
        im = im.resize(size, Image.BILINEAR)
    return im


def make_diff(before: Image.Image, after: Image.Image) -> tuple[Image.Image, float]:
    """Heatmap perbedaan (inferno) di atas backdrop rectified redup. Return (img, mean_diff_0_1)."""
    b = np.asarray(before, dtype=np.float32)
    a = np.asarray(after, dtype=np.float32)
    # magnitudo perbedaan per-piksel (rata-rata 3 kanal), 0..255
    diff = np.abs(a - b).mean(axis=2)
    mean_norm = float(diff.mean() / 255.0)
    # normalisasi persentil supaya kontras jelas (redam outlier ekstrem)
    hi = np.percentile(diff, 99.0)
    norm = np.clip(diff / (hi + 1e-6), 0, 1)
    heat = apply_inferno(norm).astype(np.float32)
    # backdrop = rectified digelapkan supaya heatmap menonjol tapi konteks tetap terlihat
    backdrop = a * 0.28
    # alpha mengikuti intensitas perbedaan (area diam -> transparan/gelap)
    alpha = (norm ** 0.7)[..., None]
    blended = heat * alpha + backdrop * (1 - alpha)
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8)), mean_norm


def scale_to_cell(im: Image.Image) -> Image.Image:
    W0, H0 = im.size
    s = CELL_W / W0
    return im.resize((CELL_W, max(1, int(round(H0 * s)))), Image.BILINEAR)


def draw_colorbar(canvas: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int):
    """Colorbar inferno kecil (rendah->tinggi) di pojok kolom diff."""
    grad = np.linspace(0, 1, h)[:, None].repeat(max(1, w), axis=1)
    bar = apply_inferno(grad[::-1])  # atas = tinggi
    return Image.fromarray(bar)


def compose(dataset_display: str, stem: str, before, after, diff, mean_pct) -> Image.Image:
    f_hdr = load_font(26, bold=True)
    f_sub = load_font(14, bold=False)
    f_cap = load_font(19, bold=True)
    f_capS = load_font(13, bold=False)
    f_leg = load_font(12, bold=True)

    cell_h = max(before.height, after.height, diff.height)
    total_w = MARGIN * 2 + CELL_W * 3 + GAP * 2
    total_h = HEADER_H + CAPTION_H + cell_h + MARGIN
    canvas = Image.new("RGB", (total_w, total_h), WHITE)
    d = ImageDraw.Draw(canvas)

    d.rectangle([0, 0, total_w, HEADER_H], fill=PANEL_BG)
    d.rectangle([0, HEADER_H - 2, total_w, HEADER_H], fill=(217, 226, 234))
    d.text((MARGIN, 12), f"MOWA: asli → rectified → peta perpindahan — {dataset_display}",
           fill=NAVY, font=f_hdr)
    d.text((MARGIN, 48),
           f"{stem}   ·   rata-rata perubahan piksel = {mean_pct*100:.1f}%   ·   heatmap: gelap=diam, terang=banyak bergeser",
           fill=MUTED, font=f_sub)

    cols = [
        ("BEFORE — asli", BEFORE_AC, before, "input kondisi A"),
        ("AFTER — MOWA rectified", AFTER_AC, after, "input kondisi B & B'"),
        ("DIFF — perpindahan piksel", DIFF_AC, diff, "|after − before|, colormap inferno"),
    ]
    x = MARGIN
    cap_y = HEADER_H
    for title, ac, cell, subcap in cols:
        d.rectangle([x, cap_y, x + CELL_W, cap_y + CAPTION_H], fill=ac)
        d.text((x + 12, cap_y + 6), title, fill=WHITE, font=f_cap)
        d.text((x + 12, cap_y + 32), subcap, fill=WHITE, font=f_capS)
        canvas.paste(cell, (x, cap_y + CAPTION_H))
        d.rectangle([x, cap_y + CAPTION_H, x + CELL_W, cap_y + CAPTION_H + cell.height],
                    outline=(210, 218, 226), width=1)
        x += CELL_W + GAP
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser(description="3-kolom before/after MOWA + difference heatmap.")
    ap.add_argument("--pick", nargs="*", default=[], help="dataset=stem override.")
    args = ap.parse_args()

    picks: Dict[str, Optional[str]] = dict(DEFAULT_PICK)
    for kv in args.pick:
        if "=" in kv:
            k, v = kv.split("=", 1)
            picks[k.strip()] = v.strip()

    index = build_index()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for ds_id, ds in index.items():
        entries = ds["entries"]
        want = picks.get(ds_id)
        entry = None
        if want:
            entry = next((e for e in entries if e["stem"] == want), None)
        if entry is None:
            entry = next((e for e in entries if e["orig_img"] and e["rect_img"].exists()), None)
        if entry is None:
            print(f"[{ds_id}] tak ada pasangan asli/rectified — lewati.")
            continue

        before_full = load_rgb(entry["orig_img"])
        # samakan dimensi ke gambar asli agar diff piksel-per-piksel valid
        after_full = load_rgb(entry["rect_img"], size=before_full.size)
        diff_full, mean_norm = make_diff(before_full, after_full)

        before = scale_to_cell(before_full)
        after = scale_to_cell(after_full)
        diff = scale_to_cell(diff_full)

        panel = compose(ds["display"], entry["stem"], before, after, diff, mean_norm)
        out_path = OUT_DIR / f"{ds_id}_diff_heatmap.png"
        panel.save(out_path)
        print(f"[{ds_id}] {entry['stem']}  mean_diff={mean_norm*100:.1f}%  -> {out_path.name}")

    print(f"\n[selesai] output di {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
