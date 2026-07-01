"""
compare_condition_panels.py — Panel pembanding visual kondisi A / B / B' untuk tiap gambar.

Menghasilkan, untuk SETIAP gambar di tiga dataset, satu panel 3-kolom berdampingan:
  A  — baseline  : bobot baseline  + gambar ASLI       + GT asli
  B  — MOWA      : bobot baseline  + gambar RECTIFIED   + GT warp
  B' — fine-tune : bobot fine-tune + gambar RECTIFIED   + GT warp

Tiap kolom menampilkan kotak PREDIKSI (oranye) + GROUND-TRUTH (hijau) beserta
jumlah deteksi dan jumlah GT. Selain panel per-gambar, dibuat pula satu
"grid ringkasan" per dataset (contact sheet dari sampel merata) untuk lihat sekilas.

Pakai .venv-yolo. Contoh:
  .venv-yolo/Scripts/python.exe src/compare_condition_panels.py
  .venv-yolo/Scripts/python.exe src/compare_condition_panels.py --datasets pio_val --limit 30
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Kurangi fragmentasi VRAM pada kartu 8GB (harus diset sebelum torch/cuda pertama kali dipakai).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

BASELINE_WEIGHTS = ROOT / "train model" / "runs_compare" / "cmp_yolov8m" / "weights" / "best.pt"
FINETUNE_WEIGHTS = ROOT / "train model" / "runs_rectified" / "ft_rectified_yolov8m" / "weights" / "best.pt"
RECT_ROOT = ROOT / "data" / "rectified"
OUT_ROOT = ROOT / "reports" / "condition_panels"

# ---- palet (selaras deck: navy/teal/amber) ----
NAVY = (15, 42, 67)
INK = (30, 41, 59)
MUTED = (100, 116, 139)
WHITE = (255, 255, 255)
PANEL_BG = (245, 248, 250)
GT_COLOR = (34, 197, 94)        # hijau — ground-truth
PRED_COLOR = (233, 138, 25)     # oranye — prediksi
ACCENT = {"A": (42, 157, 143), "B": (233, 162, 59), "B'": (28, 114, 147)}

CELL_W = 620          # lebar tiap kolom gambar (px)
GAP = 18
MARGIN = 22
HEADER_H = 74
CAPTION_H = 58
LEGEND_H = 30

CONF = 0.25
IMGSZ = 960


# --------------------------------------------------------------------------------------
# Definisi dataset & pemetaan basename -> (gambar asli, label asli)
# --------------------------------------------------------------------------------------
def build_index() -> Dict[str, Dict]:
    """Untuk tiap dataset kembalikan daftar entri {name, orig_img, orig_lbl, rect_img, rect_lbl}."""
    datasets: Dict[str, Dict] = {}

    def collect(rect_sub: str, orig_img_dirs: List[Path], orig_lbl_dirs: List[Path], display: str):
        rect_img_dir = RECT_ROOT / rect_sub / "images"
        rect_lbl_dir = RECT_ROOT / rect_sub / "labels"
        # peta basename(stem) -> path asli
        orig_img_map: Dict[str, Path] = {}
        for d in orig_img_dirs:
            if d.is_dir():
                for p in d.iterdir():
                    if p.suffix.lower() in IMAGE_EXTS:
                        orig_img_map.setdefault(p.stem, p)
        orig_lbl_map: Dict[str, Path] = {}
        for d in orig_lbl_dirs:
            if d.is_dir():
                for p in d.iterdir():
                    if p.suffix.lower() == ".txt":
                        orig_lbl_map.setdefault(p.stem, p)
        entries = []
        for rect_img in sorted(rect_img_dir.iterdir()):
            if rect_img.suffix.lower() not in IMAGE_EXTS:
                continue
            stem = rect_img.stem
            entries.append({
                "name": rect_img.name,
                "stem": stem,
                "orig_img": orig_img_map.get(stem),
                "orig_lbl": orig_lbl_map.get(stem),
                "rect_img": rect_img,
                "rect_lbl": rect_lbl_dir / (stem + ".txt"),
            })
        return {"display": display, "entries": entries}

    datasets["pio_val"] = collect(
        "pio_val",
        [ROOT / "data" / "images" / "val"],
        [ROOT / "data" / "labels" / "val"],
        "PIO val (in-domain)",
    )
    datasets["broiler_instance_seg"] = collect(
        "broiler_instance_seg",
        [ROOT / "data" / "external" / "broiler_instance_seg" / "train" / "images"],
        [ROOT / "data" / "external" / "broiler_instance_seg" / "train" / "labels"],
        "Roboflow broiler_instance_seg (external)",
    )
    fum = ROOT / "data" / "external" / "chicken_detection_fum"
    datasets["chicken_detection_fum"] = collect(
        "chicken_detection_fum",
        [fum / "test" / "images", fum / "valid" / "images", fum / "train" / "images"],
        [fum / "test" / "labels", fum / "valid" / "labels", fum / "train" / "labels"],
        "Roboflow chicken_detection_fum (external)",
    )
    return datasets


# --------------------------------------------------------------------------------------
# Inferensi
# --------------------------------------------------------------------------------------
def run_predict(model, paths: List[Path], device: str, cache_every: int = 16
                ) -> List[List[Tuple[float, float, float, float, float]]]:
    """Kembalikan list-per-gambar berisi (x1,y1,x2,y2,conf) dalam koordinat piksel gambar asli.

    Diproses SATU gambar per panggilan (batch=1) supaya tiap gambar mendapat anggaran waktu
    NMS sendiri — gambar PIO sangat padat: kalau di-batch, satu gambar padat bisa menghabiskan
    anggaran NMS batch dan memaksa gambar lain di batch itu kembali 0 deteksi. VRAM cache
    dibersihkan tiap `cache_every` gambar agar tidak OOM di kartu 8GB.
    """
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        torch = None
        has_cuda = False

    collected: Dict[str, List] = {}
    valid = [p for p in paths if p is not None and p.exists()]
    for i, p in enumerate(valid):
        results = model.predict(
            source=str(p), stream=False, imgsz=IMGSZ, conf=CONF,
            device=device, verbose=False, max_det=1000,
        )
        r = results[0]
        boxes = []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c in zip(xyxy, conf):
                boxes.append((float(x1), float(y1), float(x2), float(y2), float(c)))
        collected[str(p)] = boxes
        del results, r
        if has_cuda and torch is not None and (i + 1) % cache_every == 0:
            torch.cuda.empty_cache()

    return [collected.get(str(p), []) if p is not None else [] for p in paths]


# --------------------------------------------------------------------------------------
# Gambar / panel
# --------------------------------------------------------------------------------------
def load_font(size: int, bold: bool = False):
    candidates = (["arialbd.ttf", "C:/Windows/Fonts/arialbd.ttf"] if bold
                  else ["arial.ttf", "C:/Windows/Fonts/arial.ttf"])
    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except Exception:
            continue
    return ImageFont.load_default()


F_HDR = None
F_SUB = None
F_CAP = None
F_CAP_S = None
F_LEG = None


def init_fonts():
    global F_HDR, F_SUB, F_CAP, F_CAP_S, F_LEG
    F_HDR = load_font(26, bold=True)
    F_SUB = load_font(15, bold=False)
    F_CAP = load_font(19, bold=True)
    F_CAP_S = load_font(14, bold=False)
    F_LEG = load_font(14, bold=True)


def read_yolo(path: Optional[Path]) -> List[Tuple[float, float, float, float]]:
    rows = []
    if path is None or not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) == 5:
            _, cx, cy, w, h = parts
            rows.append((float(cx), float(cy), float(w), float(h)))
    return rows


def render_cell(img_path: Optional[Path],
                preds: List[Tuple[float, float, float, float, float]],
                gts: List[Tuple[float, float, float, float]]) -> Tuple[Image.Image, int, int]:
    """Render satu sel: gambar diskala ke CELL_W + kotak GT & prediksi. Return (img, n_pred, n_gt)."""
    if img_path is None or not img_path.exists():
        ph = Image.new("RGB", (CELL_W, int(CELL_W * 0.6)), (230, 232, 235))
        d = ImageDraw.Draw(ph)
        d.text((CELL_W // 2 - 60, int(CELL_W * 0.3)), "(tidak ada)", fill=MUTED, font=F_CAP_S)
        return ph, 0, 0

    im = Image.open(img_path).convert("RGB")
    W0, H0 = im.size
    scale = CELL_W / W0
    cell_h = max(1, int(round(H0 * scale)))
    im = im.resize((CELL_W, cell_h), Image.BILINEAR)
    d = ImageDraw.Draw(im, "RGBA")

    # ground-truth (hijau) — digambar dulu agar prediksi di atasnya
    for cx, cy, w, h in gts:
        x1 = (cx - w / 2) * W0 * scale
        y1 = (cy - h / 2) * H0 * scale
        x2 = (cx + w / 2) * W0 * scale
        y2 = (cy + h / 2) * H0 * scale
        d.rectangle([x1, y1, x2, y2], outline=GT_COLOR + (255,), width=2)

    # prediksi (oranye)
    for x1, y1, x2, y2, _c in preds:
        d.rectangle([x1 * scale, y1 * scale, x2 * scale, y2 * scale],
                    outline=PRED_COLOR + (255,), width=3)

    return im, len(preds), len(gts)


def compose_panel(entry: Dict,
                  cells: List[Tuple[Image.Image, int, int]],
                  dataset_display: str) -> Image.Image:
    labels = ["A", "B", "B'"]
    sub = ["baseline · asli", "MOWA · rectified", "fine-tune · rectified"]
    cell_h = max(c[0].height for c in cells)
    total_w = MARGIN * 2 + CELL_W * 3 + GAP * 2
    total_h = HEADER_H + CAPTION_H + cell_h + MARGIN
    canvas = Image.new("RGB", (total_w, total_h), WHITE)
    d = ImageDraw.Draw(canvas)

    # header strip
    d.rectangle([0, 0, total_w, HEADER_H], fill=PANEL_BG)
    d.rectangle([0, HEADER_H - 2, total_w, HEADER_H], fill=(217, 226, 234))
    d.text((MARGIN, 12), dataset_display, fill=NAVY, font=F_HDR)
    d.text((MARGIN, 46), entry["name"], fill=MUTED, font=F_SUB)
    # legend (kanan)
    lx = total_w - MARGIN - 300
    d.rectangle([lx, 20, lx + 16, 36], outline=PRED_COLOR, width=3)
    d.text((lx + 22, 18), "Prediksi", fill=INK, font=F_LEG)
    d.rectangle([lx + 130, 20, lx + 146, 36], outline=GT_COLOR, width=3)
    d.text((lx + 152, 18), "Ground-truth", fill=INK, font=F_LEG)

    x = MARGIN
    cap_y = HEADER_H
    for (cell_img, n_pred, n_gt), lab, sb in zip(cells, labels, sub):
        ac = ACCENT[lab]
        # caption bar
        d.rectangle([x, cap_y, x + CELL_W, cap_y + CAPTION_H], fill=ac)
        d.text((x + 12, cap_y + 6), f"{lab} — {sb}", fill=WHITE, font=F_CAP)
        d.text((x + 12, cap_y + 32), f"Prediksi: {n_pred}    GT: {n_gt}", fill=WHITE, font=F_CAP_S)
        # image
        canvas.paste(cell_img, (x, cap_y + CAPTION_H))
        # border around image
        d.rectangle([x, cap_y + CAPTION_H, x + CELL_W, cap_y + CAPTION_H + cell_img.height],
                    outline=(210, 218, 226), width=1)
        x += CELL_W + GAP
    return canvas


def build_summary_grid(dataset_id: str, dataset_display: str, panel_paths: List[Path],
                       out_path: Path, sample: int = 18, cols: int = 2) -> Optional[Path]:
    if not panel_paths:
        return None
    n = min(sample, len(panel_paths))
    # ambil sampel merata
    if len(panel_paths) <= n:
        chosen = panel_paths
    else:
        step = len(panel_paths) / n
        chosen = [panel_paths[int(i * step)] for i in range(n)]
    thumb_w = 900
    thumbs = []
    for p in chosen:
        im = Image.open(p).convert("RGB")
        s = thumb_w / im.width
        thumbs.append(im.resize((thumb_w, max(1, int(im.height * s))), Image.BILINEAR))
    rows = (len(thumbs) + cols - 1) // cols
    row_h = max(t.height for t in thumbs)
    pad = 16
    title_h = 70
    grid_w = pad + cols * (thumb_w + pad)
    grid_h = title_h + rows * (row_h + pad) + pad
    canvas = Image.new("RGB", (grid_w, grid_h), WHITE)
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, grid_w, title_h], fill=NAVY)
    d.text((pad, 12), f"Ringkasan panel A/B/B' — {dataset_display}", fill=WHITE, font=F_HDR)
    d.text((pad, 44), f"Sampel merata {len(thumbs)} dari {len(panel_paths)} gambar (panel lengkap tersimpan terpisah).",
           fill=(200, 214, 226), font=F_SUB)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        x = pad + c * (thumb_w + pad)
        y = title_h + r * (row_h + pad) + pad
        canvas.paste(t, (x, y))
        d.rectangle([x, y, x + t.width, y + t.height], outline=(210, 218, 226), width=1)
    canvas.save(out_path)
    return out_path


# --------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Panel pembanding kondisi A/B/B' per gambar.")
    ap.add_argument("--datasets", nargs="*", default=["pio_val", "broiler_instance_seg", "chicken_detection_fum"])
    ap.add_argument("--limit", type=int, default=0, help="Batasi jumlah gambar per dataset (0 = semua).")
    ap.add_argument("--device", default="0")
    ap.add_argument("--grid-sample", type=int, default=18, help="Jumlah thumbnail di grid ringkasan.")
    ap.add_argument("--no-panels", action="store_true", help="Lewati panel per-gambar, hanya grid dari yang sudah ada.")
    args = ap.parse_args()

    for w in (BASELINE_WEIGHTS, FINETUNE_WEIGHTS):
        if not w.exists():
            print(f"ERROR: bobot tidak ditemukan: {w}", file=sys.stderr)
            return 2

    init_fonts()
    from ultralytics import YOLO

    print("[load] baseline:", BASELINE_WEIGHTS.name)
    base_model = YOLO(str(BASELINE_WEIGHTS))
    print("[load] finetune:", FINETUNE_WEIGHTS.name)
    ft_model = YOLO(str(FINETUNE_WEIGHTS))

    index = build_index()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for ds_id in args.datasets:
        if ds_id not in index:
            print(f"[skip] dataset tak dikenal: {ds_id}")
            continue
        ds = index[ds_id]
        entries = ds["entries"]
        if args.limit > 0:
            entries = entries[:args.limit]
        display = ds["display"]
        out_dir = OUT_ROOT / ds_id
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{ds_id}] {len(entries)} gambar -> {out_dir}")

        panel_paths: List[Path] = []
        if not args.no_panels:
            orig_imgs = [e["orig_img"] for e in entries]
            rect_imgs = [e["rect_img"] for e in entries]
            print("  [infer] A: baseline @ asli ...")
            preds_A = run_predict(base_model, orig_imgs, args.device)
            print("  [infer] B: baseline @ rectified ...")
            preds_B = run_predict(base_model, rect_imgs, args.device)
            print("  [infer] B': fine-tune @ rectified ...")
            preds_Bp = run_predict(ft_model, rect_imgs, args.device)

            for i, e in enumerate(entries):
                gt_orig = read_yolo(e["orig_lbl"])
                gt_rect = read_yolo(e["rect_lbl"])
                cellA = render_cell(e["orig_img"], preds_A[i], gt_orig)
                cellB = render_cell(e["rect_img"], preds_B[i], gt_rect)
                cellBp = render_cell(e["rect_img"], preds_Bp[i], gt_rect)
                panel = compose_panel(e, [cellA, cellB, cellBp], display)
                out_path = out_dir / (e["stem"] + ".png")
                panel.save(out_path)
                panel_paths.append(out_path)
                if (i + 1) % 25 == 0 or (i + 1) == len(entries):
                    print(f"    {i + 1}/{len(entries)} panel")
        else:
            panel_paths = sorted(out_dir.glob("*.png"))

        grid_path = OUT_ROOT / f"{ds_id}_summary_grid.png"
        gp = build_summary_grid(ds_id, display, panel_paths, grid_path, sample=args.grid_sample)
        if gp:
            print(f"  [grid] {gp.name}")

    print(f"\n[selesai] output di {OUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
