"""
pipeline_position_diagram.py — Diagram posisi MOWA dalam pipeline: PRAPROSES sebelum YOLO.

Menegaskan bahwa MOWA (rektifikasi fisheye) diterapkan SEBELUM detektor YOLO — bukan di
dalam atau sesudahnya. Alur inti:

  Gambar mentah  ->  [MOWA rektifikasi]  ->  Gambar rectified  ->  [YOLO deteksi]  ->  Output

Lalu tiga kondisi eksperimen menunjukkan di mana MOWA berada:
  A  : Gambar ->               [YOLO baseline]   -> output    (TANPA MOWA)
  B  : Gambar -> [MOWA] ->      [YOLO baseline]   -> output
  B' : Gambar -> [MOWA] ->      [YOLO fine-tune]  -> output

Pakai .venv-yolo (Pillow). Output: reports/diagrams/pipeline_position_mowa.png
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from compare_condition_panels import load_font

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "diagrams"
ORIG_IMG = ROOT / "data" / "images" / "val" / "C-W3-V0007.jpg"
RECT_IMG = ROOT / "data" / "rectified" / "pio_val" / "images" / "C-W3-V0007.jpg"

NAVY = (15, 42, 67)
NAVY2 = (23, 58, 90)
INK = (30, 41, 59)
MUTED = (95, 109, 125)
WHITE = (255, 255, 255)
LIGHT = (245, 248, 250)
LINE = (210, 218, 226)
TEAL = (28, 114, 147)
TEAL2 = (42, 157, 143)
AMBER = (233, 162, 59)
RAW = (120, 132, 148)     # abu — gambar mentah/generic
ARROW = (60, 80, 100)

W = 2000
H = 1180
MARGIN = 36


def thumb(path: Path, tw: int) -> Image.Image:
    im = Image.open(path).convert("RGB")
    s = tw / im.width
    return im.resize((tw, max(1, int(im.height * s))), Image.BILINEAR)


def rrect(d, box, r, fill=None, outline=None, width=1):
    d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


def arrow(d, x1, y, x2, color=ARROW, w=6):
    """Panah horizontal dari x1 ke x2 pada tinggi y."""
    d.line([(x1, y), (x2 - 12, y)], fill=color, width=w)
    d.polygon([(x2, y), (x2 - 16, y - 10), (x2 - 16, y + 10)], fill=color)


def main() -> int:
    canvas = Image.new("RGB", (W, H), LIGHT)
    d = ImageDraw.Draw(canvas)

    f_title = load_font(34, bold=True)
    f_sub = load_font(17, bold=False)
    f_stage = load_font(21, bold=True)
    f_stageS = load_font(14, bold=False)
    f_tag = load_font(15, bold=True)
    f_h2 = load_font(20, bold=True)
    f_cond = load_font(22, bold=True)
    f_condS = load_font(14, bold=False)
    f_note = load_font(14, bold=False)
    f_badge = load_font(13, bold=True)

    # ---- header ----
    d.rectangle([0, 0, W, 92], fill=NAVY)
    d.text((MARGIN, 16), "Posisi MOWA dalam Pipeline — PRAPROSES, Sebelum YOLO",
           fill=WHITE, font=f_title)
    d.text((MARGIN, 60),
           "MOWA merektifikasi gambar terlebih dahulu; hasilnya baru dimasukkan ke detektor YOLO. "
           "MOWA tidak berada di dalam atau sesudah YOLO.",
           fill=(200, 214, 226), font=f_sub)

    # ================= ALUR INTI =================
    d.text((MARGIN, 116), "ALUR INTI PEMROSESAN", fill=TEAL, font=f_h2)

    flow_y = 168
    flow_h = 300
    tw = 300  # thumb width
    th = int(tw * 9 / 16)

    # posisi 5 stage
    n = 5
    gap = 70
    stage_w = (W - 2 * MARGIN - (n - 1) * gap) // n
    xs = [MARGIN + i * (stage_w + gap) for i in range(n)]
    mid_y = flow_y + flow_h // 2

    def stage_card(i, fill, outline):
        x = xs[i]
        rrect(d, [x, flow_y, x + stage_w, flow_y + flow_h], 14, fill=fill, outline=outline, width=2)
        return x

    # 1) Gambar mentah (thumb asli)
    x = stage_card(0, WHITE, LINE)
    d.rectangle([x, flow_y, x + stage_w, flow_y + 40], fill=RAW)
    d.text((x + 14, flow_y + 9), "1 · Gambar mentah", fill=WHITE, font=f_stage)
    t = thumb(ORIG_IMG, stage_w - 28)
    canvas.paste(t, (x + 14, flow_y + 52))
    d.rectangle([x + 14, flow_y + 52, x + 14 + t.width, flow_y + 52 + t.height], outline=LINE, width=1)
    d.text((x + 14, flow_y + 52 + t.height + 10),
           "kamera atas peternakan;", fill=INK, font=f_stageS)
    d.text((x + 14, flow_y + 52 + t.height + 30),
           "ada distorsi barrel/fisheye", fill=INK, font=f_stageS)

    # 2) MOWA (praproses) — disorot
    x = stage_card(1, (232, 244, 246), TEAL)
    d.rectangle([x, flow_y, x + stage_w, flow_y + 40], fill=TEAL)
    d.text((x + 14, flow_y + 9), "2 · MOWA", fill=WHITE, font=f_stage)
    # ikon-ish: lingkaran + teks
    cx = x + stage_w // 2
    d.ellipse([cx - 46, flow_y + 78, cx + 46, flow_y + 170], outline=TEAL, width=5)
    d.text((cx - 38, flow_y + 108), "warp", fill=TEAL, font=f_stage)
    d.text((x + 16, flow_y + 190), "Model rektifikasi terpelajar", fill=INK, font=f_stageS)
    d.text((x + 16, flow_y + 210), "(TPAMI 2025). Meluruskan", fill=INK, font=f_stageS)
    d.text((x + 16, flow_y + 230), "distorsi TANPA kalibrasi", fill=INK, font=f_stageS)
    d.text((x + 16, flow_y + 250), "kamera.", fill=INK, font=f_stageS)

    # 3) Gambar rectified (thumb rectified)
    x = stage_card(2, WHITE, LINE)
    d.rectangle([x, flow_y, x + stage_w, flow_y + 40], fill=TEAL2)
    d.text((x + 14, flow_y + 9), "3 · Gambar rectified", fill=WHITE, font=f_stage)
    t = thumb(RECT_IMG, stage_w - 28)
    canvas.paste(t, (x + 14, flow_y + 52))
    d.rectangle([x + 14, flow_y + 52, x + 14 + t.width, flow_y + 52 + t.height], outline=LINE, width=1)
    d.text((x + 14, flow_y + 52 + t.height + 10),
           "garis lurus, tepi ditarik;", fill=INK, font=f_stageS)
    d.text((x + 14, flow_y + 52 + t.height + 30),
           "inilah input ke YOLO", fill=INK, font=f_stageS)

    # 4) YOLO
    x = stage_card(3, (245, 240, 230), AMBER)
    d.rectangle([x, flow_y, x + stage_w, flow_y + 40], fill=AMBER)
    d.text((x + 14, flow_y + 9), "4 · YOLO deteksi", fill=NAVY, font=f_stage)
    cx = x + stage_w // 2
    d.rounded_rectangle([cx - 60, flow_y + 88, cx + 60, flow_y + 150], 10, outline=AMBER, width=5)
    d.text((cx - 44, flow_y + 104), "YOLOv8m", fill=(150, 100, 20), font=f_stage)
    d.text((x + 16, flow_y + 172), "Detektor broiler terlatih", fill=INK, font=f_stageS)
    d.text((x + 16, flow_y + 192), "pada PIO. Menghasilkan", fill=INK, font=f_stageS)
    d.text((x + 16, flow_y + 212), "kotak (bbox) tiap ayam.", fill=INK, font=f_stageS)

    # 5) Output
    x = stage_card(4, WHITE, LINE)
    d.rectangle([x, flow_y, x + stage_w, flow_y + 40], fill=NAVY)
    d.text((x + 14, flow_y + 9), "5 · Output", fill=WHITE, font=f_stage)
    items = [
        ("Bounding box", "lokasi & jumlah broiler"),
        ("Estimasi berat", "geometri bbox -> Cobb500"),
        ("Deteksi anomali", "voting ensemble"),
    ]
    yy = flow_y + 58
    for h, s in items:
        d.ellipse([x + 16, yy + 4, x + 28, yy + 16], fill=TEAL)
        d.text((x + 38, yy), h, fill=NAVY, font=f_tag)
        d.text((x + 38, yy + 22), s, fill=MUTED, font=f_stageS)
        yy += 62

    # panah antar stage
    for i in range(n - 1):
        ax1 = xs[i] + stage_w
        ax2 = xs[i + 1]
        arrow(d, ax1 + 8, mid_y, ax2 - 6)

    # banner "PRAPROSES sebelum YOLO" di atas MOWA -> menunjuk stage 2
    bx = xs[1]
    d.rounded_rectangle([bx - 6, flow_y - 40, bx + stage_w + 6, flow_y - 8], 8, fill=TEAL)
    d.text((bx + 10, flow_y - 36), "◄ PRAPROSES — diterapkan SEBELUM YOLO ►", fill=WHITE, font=f_badge)

    # ================= TIGA KONDISI =================
    sec_y = flow_y + flow_h + 56
    d.text((MARGIN, sec_y), "DI MANA MOWA BERADA PADA TIAP KONDISI", fill=TEAL, font=f_h2)

    cond_y = sec_y + 40
    row_h = 128
    row_gap = 18
    # blok mini untuk tiap tahap dalam baris kondisi
    def mini(x, y, w, h, text, fill, txtcol, sub=None, dashed=False):
        if dashed:
            # kotak putus-putus (menandai MOWA absen)
            for xx in range(int(x), int(x + w), 14):
                d.line([(xx, y), (min(xx + 7, x + w), y)], fill=fill, width=2)
                d.line([(xx, y + h), (min(xx + 7, x + w), y + h)], fill=fill, width=2)
            for yy2 in range(int(y), int(y + h), 14):
                d.line([(x, yy2), (x, min(yy2 + 7, y + h))], fill=fill, width=2)
                d.line([(x + w, yy2), (x + w, min(yy2 + 7, y + h))], fill=fill, width=2)
        else:
            rrect(d, [x, y, x + w, y + h], 10, fill=fill, outline=None)
        tb = d.textbbox((0, 0), text, font=f_cond)
        d.text((x + (w - (tb[2] - tb[0])) / 2, y + (h - (tb[3] - tb[1])) / 2 - (8 if sub else 0)),
               text, fill=txtcol, font=f_cond)
        if sub:
            tb2 = d.textbbox((0, 0), sub, font=f_condS)
            d.text((x + (w - (tb2[2] - tb2[0])) / 2, y + h / 2 + 8), sub, fill=txtcol, font=f_condS)

    conds = [
        ("A", (42, 157, 143), False, "baseline", "TANPA MOWA"),
        ("B", AMBER, True, "baseline", "dengan MOWA"),
        ("B'", TEAL, True, "fine-tune", "dengan MOWA"),
    ]
    label_w = 96
    img_w = 150
    box_w = 180
    aw = 44  # arrow span

    for r, (lab, col, has_mowa, yolo_kind, tag) in enumerate(conds):
        y = cond_y + r * (row_h + row_gap)
        # kartu baris
        rrect(d, [MARGIN, y, W - MARGIN, y + row_h], 12, fill=WHITE, outline=LINE, width=1)
        rrect(d, [MARGIN, y, MARGIN + 8, y + row_h], 12, fill=col)
        # badge kondisi
        rrect(d, [MARGIN + 22, y + 30, MARGIN + 22 + 66, y + row_h - 30], 10, fill=col)
        tb = d.textbbox((0, 0), lab, font=load_font(30, bold=True))
        d.text((MARGIN + 22 + (66 - (tb[2] - tb[0])) / 2, y + row_h / 2 - 20), lab,
               fill=(NAVY if lab == "B" else WHITE), font=load_font(30, bold=True))
        d.text((MARGIN + 20, y + row_h - 26), tag, fill=MUTED, font=f_note)

        cy = y + row_h / 2
        x = MARGIN + 120
        # 1) Gambar
        mini(x, cy - 34, img_w, 68, "Gambar", RAW, WHITE, sub="mentah")
        x += img_w
        arrow(d, x + 6, cy, x + aw)
        x += aw + 6
        # 2) MOWA (atau kosong/dashed)
        if has_mowa:
            mini(x, cy - 34, box_w, 68, "MOWA", TEAL, WHITE, sub="rektifikasi")
        else:
            mini(x, cy - 34, box_w, 68, "(tanpa MOWA)", LINE, MUTED, dashed=True)
        x += box_w
        arrow(d, x + 6, cy, x + aw)
        x += aw + 6
        # 3) YOLO
        ycol = AMBER if yolo_kind == "baseline" else TEAL
        mini(x, cy - 34, box_w, 68, "YOLO", ycol, (NAVY if ycol == AMBER else WHITE), sub=yolo_kind)
        x += box_w
        arrow(d, x + 6, cy, x + aw)
        x += aw + 6
        # 4) Output
        mini(x, cy - 34, img_w, 68, "Output", NAVY, WHITE, sub="bbox+anomali")

    # ================= CATATAN =================
    note_y = cond_y + 3 * (row_h + row_gap) + 12
    rrect(d, [MARGIN, note_y, W - MARGIN, note_y + 78], 10, fill=(240, 244, 248), outline=LINE, width=1)
    d.rectangle([MARGIN, note_y, MARGIN + 8, note_y + 78], fill=TEAL)
    d.text((MARGIN + 22, note_y + 12),
           "Catatan: MOWA selalu berupa PRAPROSES (input-side). Kondisi A tidak memakai MOWA "
           "(acuan). B & B' memakai gambar rectified yang IDENTIK —",
           fill=INK, font=f_note)
    d.text((MARGIN + 22, note_y + 40),
           "yang membedakan hanya bobot YOLO: B pakai detektor baseline (domain mismatch), "
           "B' pakai detektor yang di-fine-tune pada domain rectified.",
           fill=INK, font=f_note)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "pipeline_position_mowa.png"
    canvas.save(out)
    print(f"[diagram] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
