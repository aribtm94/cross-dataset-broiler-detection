"""
mowa_before_after.py — Perbandingan BEFORE/AFTER rektifikasi MOWA (1 gambar per dataset).

Menampilkan gambar ASLI vs hasil MOWA berdampingan, dengan overlay grid LURUS pada
keduanya supaya efek debarreling/pelurusan terlihat: garis/rel yang melengkung di
gambar asli menjadi lurus & sejajar grid setelah MOWA.

Catatan: gambar rectified untuk kondisi B dan B' IDENTIK (output MOWA sama); yang beda
hanya bobot detektor. Jadi cukup satu perbandingan before/after per dataset.

Tidak butuh GPU / inferensi — hanya baca gambar. Pakai .venv-yolo (ada Pillow) atau
python mana pun yang punya Pillow.

Contoh:
  .venv-yolo/Scripts/python.exe src/mowa_before_after.py
  .venv-yolo/Scripts/python.exe src/mowa_before_after.py --pick pio_val=C-W1-V0005
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

from PIL import Image, ImageDraw, ImageFont

from compare_condition_panels import build_index, load_font  # reuse

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "mowa_before_after"

NAVY = (15, 42, 67)
INK = (30, 41, 59)
MUTED = (100, 116, 139)
WHITE = (255, 255, 255)
PANEL_BG = (245, 248, 250)
GRID = (0, 200, 255)          # cyan — grid referensi lurus
BEFORE_AC = (192, 87, 70)     # merah bata — "asli/terdistorsi"
AFTER_AC = (42, 157, 143)     # teal — "MOWA rectified"

CELL_W = 780
GAP = 20
MARGIN = 24
HEADER_H = 84
CAPTION_H = 56
GRID_N = 12                   # jumlah kolom grid (baris menyesuaikan rasio)

# gambar default per dataset (dipilih yang distorsi barrel-nya jelas)
DEFAULT_PICK = {
    "pio_val": "C-W1-V0001",
    "broiler_instance_seg": None,          # None -> ambil entri pertama
    "chicken_detection_fum": None,
}


def draw_grid(im: Image.Image, n_cols: int = GRID_N) -> None:
    d = ImageDraw.Draw(im, "RGBA")
    w, h = im.size
    step = w / n_cols
    x = step
    while x < w:
        d.line([(x, 0), (x, h)], fill=GRID + (110,), width=1)
        x += step
    y = step
    while y < h:
        d.line([(0, y), (w, y)], fill=GRID + (110,), width=1)
        y += step


def render_cell(img_path: Path) -> Image.Image:
    im = Image.open(img_path).convert("RGB")
    W0, H0 = im.size
    scale = CELL_W / W0
    cell_h = max(1, int(round(H0 * scale)))
    im = im.resize((CELL_W, cell_h), Image.BILINEAR)
    draw_grid(im)
    return im


def compose(dataset_display: str, stem: str,
            before: Image.Image, after: Image.Image) -> Image.Image:
    f_hdr = load_font(28, bold=True)
    f_sub = load_font(15, bold=False)
    f_cap = load_font(20, bold=True)
    f_capS = load_font(14, bold=False)
    f_leg = load_font(14, bold=True)

    cell_h = max(before.height, after.height)
    total_w = MARGIN * 2 + CELL_W * 2 + GAP
    total_h = HEADER_H + CAPTION_H + cell_h + MARGIN
    canvas = Image.new("RGB", (total_w, total_h), WHITE)
    d = ImageDraw.Draw(canvas)

    # header
    d.rectangle([0, 0, total_w, HEADER_H], fill=PANEL_BG)
    d.rectangle([0, HEADER_H - 2, total_w, HEADER_H], fill=(217, 226, 234))
    d.text((MARGIN, 12), f"Rektifikasi MOWA — {dataset_display}", fill=NAVY, font=f_hdr)
    d.text((MARGIN, 50), f"{stem}   ·   grid cyan = garis referensi lurus (before=melengkung, after=lurus)",
           fill=MUTED, font=f_sub)
    # legend
    lx = total_w - MARGIN - 150
    d.line([(lx, 30), (lx + 24, 30)], fill=GRID, width=2)
    d.text((lx + 32, 22), "Grid lurus", fill=INK, font=f_leg)

    cols = [("BEFORE — asli (terdistorsi)", BEFORE_AC, before),
            ("AFTER — MOWA rectified (B & B')", AFTER_AC, after)]
    x = MARGIN
    cap_y = HEADER_H
    for title, ac, cell in cols:
        d.rectangle([x, cap_y, x + CELL_W, cap_y + CAPTION_H], fill=ac)
        d.text((x + 12, cap_y + 8), title, fill=WHITE, font=f_cap)
        d.text((x + 12, cap_y + 34), "gambar sama dipakai kondisi B dan B'" if ac == AFTER_AC
               else "input detektor kondisi A", fill=WHITE, font=f_capS)
        canvas.paste(cell, (x, cap_y + CAPTION_H))
        d.rectangle([x, cap_y + CAPTION_H, x + CELL_W, cap_y + CAPTION_H + cell.height],
                    outline=(210, 218, 226), width=1)
        x += CELL_W + GAP
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser(description="Before/after rektifikasi MOWA per dataset.")
    ap.add_argument("--pick", nargs="*", default=[],
                    help="Override pilihan gambar: dataset=stem (mis. pio_val=C-W1-V0005).")
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
                print(f"[{ds_id}] stem '{want}' tak ditemukan, pakai entri pertama.")
        if entry is None:
            # entri pertama yang punya gambar asli + rectified
            entry = next((e for e in entries if e["orig_img"] and e["rect_img"].exists()), None)
        if entry is None:
            print(f"[{ds_id}] tak ada pasangan asli/rectified — lewati.")
            continue

        before = render_cell(entry["orig_img"])
        after = render_cell(entry["rect_img"])
        panel = compose(ds["display"], entry["stem"], before, after)
        out_path = OUT_DIR / f"{ds_id}_before_after.png"
        panel.save(out_path)
        print(f"[{ds_id}] {entry['stem']} -> {out_path.name}")

    print(f"\n[selesai] output di {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
