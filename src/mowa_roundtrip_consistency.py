"""
mowa_roundtrip_consistency.py — Metrik ROUND-TRIP (bolak-balik) rektifikasi MOWA.

Tujuan (skripsi):
  Mengkuantifikasi KERUGIAN INFORMASI akibat rektifikasi MOWA lewat uji bolak-balik
  FORWARD -> INVERSE. Untuk tiap gambar X:
      X --(MOWA rectify, forward)--> Y --(inverse-warp)--> X'
  lalu diukur seberapa jauh X' menyimpang dari X. Kalau rektifikasi benar-benar
  reversibel (tak ada informasi hilang), X' == X. Semakin besar galat rekonstruksi
  + semakin banyak "hole" (piksel tak-terisi) + semakin banyak konten yang terdorong
  keluar frame, semakin kuat bukti OBJEKTIF bahwa MOWA membuang informasi — argumen
  utama mengapa MOWA menurunkan mAP pada kondisi B (lihat project-mowa-ab-result).

  Ini BUKAN re-rektifikasi iteratif. Konsep kanonis: inverse/backward warping diukur
  lewat round-trip / cycle-consistency reconstruction error:
    - CycleMorph (Kim dkk., arXiv:2008.05772) — cycle-consistency pada registrasi.
    - Sánchez dkk., "Computing Inverse Optical Flow" (Pattern Recognition Letters, 2015)
      — iterasi fixed-point untuk membalik medan aliran (dipakai di sini).
    - Inverse Consistency Error / ICE (Christiansen & Johnson, 2001).

  MOWA bersifat FORWARD-ONLY (tak punya inverse bawaan), jadi langkah inverse di sini
  dibangun secara NUMERIK dari medan warp MOWA sendiri -> hanya APROKSIMASI, bukan
  invers analitik. Ini dinyatakan eksplisit demi kejujuran ilmiah.

Metode (inti — dokumentasikan presisi):
  1) Medan MOWA (dari compute_flows) = dua tahap BACKWARD map (tps2flow lalu flow),
     di mana untuk tiap piksel OUTPUT, (identitas + medan) menunjuk lokasi SAMPEL di
     citra INPUT. Forward Y = _apply_full_warp(X, tps2flow, flow).
  2) Kedua tahap digabung jadi SATU medan backward D (lihat _combine_backward_field):
        Y(p) = X( p + flow(p) + tps2flow(p + flow(p)) )
        => D(p) = flow(p) + tps2flow(p + flow(p)) = flow + _resample_xy(tps2flow, flow)
     Jadi koordinat sumber gabungan C(p) = p + D(p), dan Y(p) = X(C(p)).
  3) INVERSE via FIXED-POINT (Sánchez dkk.): cari medan g sehingga X'(q) = Y(q + g(q))
     merekonstruksi X. Karena p + D(p) = q, berlaku g(q) = -D(q + g(q)); diiterasi:
        g <- -_resample_xy(D, g)      (3-5 iterasi cukup untuk medan mulus MOWA)
     lalu X' = _resample_xy(Y, g, 'bilinear'). Tetap di mesin grid_sample yang sama.

Metrik per gambar (area valid = piksel X' yang benar-benar terisi, hole dikecualikan):
  - recon_mae      : rata-rata |X - X'| pada area valid, satuan 0-255.
  - recon_psnr     : PSNR(X, X') pada area valid (max=255), satuan dB.
  - hole_rate      : fraksi piksel X' tak-terisi (inverse-nya menunjuk keluar frame).
  - edge_loss_rate : fraksi piksel OUTPUT forward yang sumbernya (identitas + D) keluar
                     frame [0,W)x[0,H) -> konten terdorong keluar pandang saat rektifikasi,
                     TAK bisa dikembalikan (unrecoverable).
Agregat per-dataset = rata-rata tiap metrik; ditambah agregat "overall" lintas dataset.

Butuh CUDA (utils_transform MOWA meng-hardcode .cuda()). Tanpa GPU -> exit 3.

Contoh pemakaian (dari root proyek, venv khusus MOWA):
  .venv-mowa/Scripts/python.exe src/mowa_roundtrip_consistency.py --limit 30
  .venv-mowa/Scripts/python.exe src/mowa_roundtrip_consistency.py \
      --datasets pio_val,broiler_instance_seg --limit 10 --overlays 4

Output (semua di reports/roundtrip/, gitignored — jangan commit):
  <id>_metrics.csv           metrik per-gambar tiap dataset
  roundtrip_summary.json     agregat per-dataset + overall
  <id>_overlay_NN_<stem>.png ~4 sandingan [X | Y | X' | heatmap |X-X'|] per dataset
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

# Reuse mesin MOWA — JANGAN reimplementasi (impor dari modul sekandang).
from mowa_rectify import (
    INPUT_SIZE,
    add_mowa_to_path,
    build_net,
    compute_flows,
    list_images,
    load_checkpoint,
    _apply_full_warp,
    _identity_grid_xy,
    _resample_xy,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "roundtrip"
TPS_POINTS = [10, 12, 14, 16]
INVERSE_ITERS_DEFAULT = 5  # iterasi fixed-point untuk membalik medan (3-5 cukup, Sánchez dkk.)

# Definisi dataset (hardcode selaras eval_detection.py). Round-trip tak butuh label,
# jadi hanya folder-folder images yang didaftarkan (union bila >1 split).
DATASETS: List[Dict] = [
    {
        "id": "pio_val",
        "display": "PIO val (in-domain)",
        "img_dirs": [ROOT / "data" / "images" / "val"],
    },
    {
        "id": "broiler_instance_seg",
        "display": "Roboflow broiler_instance_seg (external)",
        "img_dirs": [ROOT / "data" / "external" / "broiler_instance_seg" / "train" / "images"],
    },
    {
        "id": "chicken_detection_fum",
        "display": "Roboflow chicken_detection_fum (external)",
        "img_dirs": [
            ROOT / "data" / "external" / "chicken_detection_fum" / "test" / "images",
            ROOT / "data" / "external" / "chicken_detection_fum" / "valid" / "images",
            ROOT / "data" / "external" / "chicken_detection_fum" / "train" / "images",
        ],
    },
]


# ---------------------------------------------------------------------------
# Medan backward gabungan & inversi numerik (fixed-point, Sánchez dkk. PRL 2015).
# ---------------------------------------------------------------------------

def _combine_backward_field(tps2flow: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Gabungkan dua tahap backward MOWA jadi SATU medan offset D (1,2,H,W).

    _apply_full_warp menerapkan tps2flow lalu flow, sehingga:
        Y(p) = X( p + flow(p) + tps2flow(p + flow(p)) )
    => D(p) = flow(p) + tps2flow(p + flow(p)).
    Suku kedua = tps2flow disampel di lokasi (p + flow) = _resample_xy(tps2flow, flow):
    grid_sample menginterpolasi medan offset 2-kanal itu secara bilinear (sah karena
    kedua kanal adalah nilai offset kontinu). D dipakai untuk edge_loss & inversi.
    """
    tps_at_shifted = _resample_xy(tps2flow, flow, mode="bilinear")  # tps2flow(p + flow(p))
    return flow + tps_at_shifted


def _invert_backward_field(field_d: torch.Tensor, iters: int) -> torch.Tensor:
    """Aproksimasi invers medan backward D via fixed-point (Sánchez dkk. PRL 2015).

    Diinginkan g s.d. X'(q) = Y(q + g(q)) merekonstruksi X. Karena p + D(p) = q berlaku
    g(q) = -D(q + g(q)); diiterasi g <- -_resample_xy(D, g) dari g0 = 0. Medan MOWA
    mulus -> konvergen dalam 3-5 iterasi. Bukan invers analitik (MOWA forward-only).
    """
    g = torch.zeros_like(field_d)
    for _ in range(max(1, iters)):
        g = -_resample_xy(field_d, g, mode="bilinear")
    return g


def _out_of_frame_mask(offset: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Mask (1,1,H,W) bool: True bila (identitas + offset) keluar frame [0,W-1]x[0,H-1]."""
    uv = _identity_grid_xy(h, w, offset.device) + offset  # 1,2,H,W koordinat sumber
    sx = uv[:, 0:1, :, :]
    sy = uv[:, 1:2, :, :]
    inside = (sx >= 0) & (sx <= (w - 1)) & (sy >= 0) & (sy <= (h - 1))
    return ~inside


def _to_uint8_bgr(img_t: torch.Tensor) -> np.ndarray:
    """Tensor citra (1,3,H,W) [0,1] RGB-order-input -> HxWx3 uint8 (urutan kanal apa adanya).

    Catatan: input1 dibangun langsung dari BGR OpenCV (tanpa swap), jadi urutan kanal
    tensor == BGR; kembalikan apa adanya supaya cv2.imwrite benar.
    """
    arr = img_t[0].clamp(0, 1).mul(255.0).round().cpu().numpy().transpose(1, 2, 0)
    return arr.astype(np.uint8)


# ---------------------------------------------------------------------------
# Proses satu gambar.
# ---------------------------------------------------------------------------

@torch.no_grad()
def roundtrip_one(net, img_bgr: np.ndarray, device: torch.device, use_fp16: bool,
                  tps_cap: int, inv_iters: int) -> Tuple[Dict, Dict]:
    """Round-trip satu gambar BGR uint8. Return (metrics, tensors_for_overlay).

    metrics: recon_mae, recon_psnr, hole_rate, edge_loss_rate.
    tensors_for_overlay: X, Y, Xp (uint8 BGR HxWx3) + hole mask (HxW bool) + diff (HxW f32).
    """
    ori_h, ori_w = img_bgr.shape[:2]

    input1 = np.transpose(img_bgr.astype(np.float32) / 255.0, (2, 0, 1))[None]  # 1,3,H,W (BGR)
    resized = cv2.resize(img_bgr, (INPUT_SIZE, INPUT_SIZE)).astype(np.float32) / 255.0
    input2 = np.transpose(resized, (2, 0, 1))[None]  # 1,3,256,256
    mask = np.ones((1, 1, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)

    x_t = torch.from_numpy(input1).float().to(device)
    input2_t = torch.from_numpy(input2).float().to(device)
    mask_t = torch.from_numpy(mask).float().to(device)

    # 1) Forward rectify X -> Y (dua tahap backward, sama persis dgn mowa_rectify).
    tps2flow, flow = compute_flows(net, input2_t, mask_t, ori_h, ori_w, TPS_POINTS,
                                   device, use_fp16, tps_cap=tps_cap)
    y_t = _apply_full_warp(x_t, tps2flow, flow, mode="bilinear")

    # 2) Medan backward gabungan D + edge_loss (sumber forward keluar frame).
    field_d = _combine_backward_field(tps2flow, flow)
    edge_out = _out_of_frame_mask(field_d, ori_h, ori_w)  # 1,1,H,W
    edge_loss_rate = float(edge_out.float().mean().item())

    # 3) Inverse-warp Y -> X' via fixed-point; hole = inverse menunjuk keluar frame.
    g = _invert_backward_field(field_d, inv_iters)
    hole = _out_of_frame_mask(g, ori_h, ori_w)  # 1,1,H,W
    xp_t = _resample_xy(y_t, g, mode="bilinear")

    # Nolkan piksel hole di X' agar overlay & agregasi konsisten (0 = tak-terisi).
    valid = (~hole).float()
    xp_t = xp_t * valid

    # 4) Metrik rekonstruksi pada area valid, satuan 0-255.
    x255 = x_t * 255.0
    xp255 = xp_t * 255.0
    diff = (x255 - xp255).abs()  # 1,3,H,W
    valid_px = valid.sum().item() * 3  # jumlah piksel-kanal valid (broadcast 3 kanal)
    if valid_px > 0:
        abs_sum = (diff * valid).sum().item()
        sq_sum = ((diff * valid) ** 2).sum().item()
        recon_mae = abs_sum / valid_px
        mse = sq_sum / valid_px
        recon_psnr = float("inf") if mse <= 1e-12 else 10.0 * math.log10((255.0 ** 2) / mse)
    else:
        recon_mae = float("nan")
        recon_psnr = float("nan")
    hole_rate = float(hole.float().mean().item())

    metrics = {
        "recon_mae": recon_mae,
        "recon_psnr": recon_psnr,
        "hole_rate": hole_rate,
        "edge_loss_rate": edge_loss_rate,
    }

    # Bahan overlay: diff grayscale (rata-rata 3 kanal), 0-255.
    diff_gray = diff.mean(dim=1)[0].cpu().numpy()  # H,W float 0-255
    overlay = {
        "X": _to_uint8_bgr(x_t),
        "Y": _to_uint8_bgr(y_t),
        "Xp": _to_uint8_bgr(xp_t),
        "diff_gray": diff_gray,
    }
    return metrics, overlay


def make_overlay(overlay: Dict, stem: str) -> np.ndarray:
    """Sandingan horizontal [X | Y | X' | heatmap |X-X'|] jadi satu gambar BGR uint8."""
    x = overlay["X"]
    h = x.shape[0]

    def _fit(im: np.ndarray) -> np.ndarray:
        if im.shape[0] != h:
            scale = h / im.shape[0]
            im = cv2.resize(im, (max(1, int(round(im.shape[1] * scale))), h))
        return im

    y = _fit(overlay["Y"])
    xp = _fit(overlay["Xp"])

    diff = overlay["diff_gray"]
    hi = float(np.percentile(diff, 99.0))  # redam outlier ekstrem agar kontras jelas
    norm = np.clip(diff / (hi + 1e-6), 0, 1)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    heat = _fit(heat)

    sep = np.full((h, 4, 3), 255, np.uint8)
    stacked = np.hstack([x, sep, y, sep, xp, sep, heat])

    # Label tipis di atas tiap panel untuk keterbacaan.
    labels = ["X (asli)", "Y (MOWA)", "X' (bolak-balik)", "|X - X'|"]
    panel_w = x.shape[1]
    for i, lab in enumerate(labels):
        ox = i * (panel_w + 4) + 6
        cv2.putText(stacked, lab, (ox, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(stacked, lab, (ox, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return stacked


def _aggregate(rows: List[Dict]) -> Dict:
    """Rata-rata tiap metrik (abaikan NaN/inf pada PSNR)."""
    def _mean(key: str, finite_only: bool = False) -> float:
        vals = [r[key] for r in rows]
        if finite_only:
            vals = [v for v in vals if math.isfinite(v)]
        else:
            vals = [v for v in vals if not math.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "n": len(rows),
        "recon_mae_mean": _mean("recon_mae"),
        "recon_psnr_mean": _mean("recon_psnr", finite_only=True),
        "hole_rate_mean": _mean("hole_rate"),
        "edge_loss_rate_mean": _mean("edge_loss_rate"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Metrik round-trip (bolak-balik) rektifikasi MOWA: forward->inverse.")
    ap.add_argument("--mowa-root", type=Path, default=Path("vendor/MOWA"), help="Root repo MOWA.")
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("vendor/MOWA/checkpoint/mowa_pretrained.pth"))
    ap.add_argument("--limit", type=int, default=0,
                    help="Proses N gambar pertama per dataset (0 = semua).")
    ap.add_argument("--datasets", default="",
                    help="Daftar id dataset dipisah koma (default: semua).")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tps-cap", type=int, default=384,
                    help="Sisi terpanjang komputasi TPS coarse (default 384).")
    ap.add_argument("--no-fp16", action="store_true", help="Nonaktifkan autocast fp16.")
    ap.add_argument("--overlays", type=int, default=4,
                    help="Jumlah overlay PNG per dataset (default 4).")
    ap.add_argument("--inv-iters", type=int, default=INVERSE_ITERS_DEFAULT,
                    help="Iterasi fixed-point untuk inversi medan (default 5).")
    args = ap.parse_args()

    # Guard CUDA — MOWA hardcode .cuda() di utils_transform.
    if not torch.cuda.is_available():
        print("ERROR: MOWA butuh CUDA (utils_transform.resample_image_xy hardcode .cuda()). "
              "Tidak ada GPU terdeteksi.", file=sys.stderr)
        return 3
    device = torch.device(f"cuda:{args.gpu}")

    wanted = {s.strip() for s in args.datasets.split(",") if s.strip()}
    targets = [d for d in DATASETS if not wanted or d["id"] in wanted]
    if not targets:
        print(f"ERROR: tak ada dataset cocok dgn --datasets={args.datasets!r}. "
              f"Pilihan: {[d['id'] for d in DATASETS]}", file=sys.stderr)
        return 2

    add_mowa_to_path(args.mowa_root)
    use_fp16 = not args.no_fp16
    print(f"[roundtrip] device={device} fp16={use_fp16} inv_iters={args.inv_iters}")
    net = build_net(device)
    load_checkpoint(net, args.checkpoint, device)
    net.eval()
    print(f"[roundtrip] model dimuat dari {args.checkpoint}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: Dict = {
        "method": "forward (MOWA _apply_full_warp) -> inverse fixed-point (Sánchez dkk. PRL 2015) "
                  "atas medan backward gabungan D = flow + tps2flow(p+flow); "
                  "aproksimasi numerik (MOWA forward-only, tanpa invers analitik).",
        "inv_iters": args.inv_iters,
        "tps_cap": args.tps_cap,
        "fp16": use_fp16,
        "units": {"recon_mae": "0-255", "recon_psnr": "dB (max=255)",
                  "hole_rate": "fraksi", "edge_loss_rate": "fraksi"},
        "datasets": {},
    }
    all_rows: List[Dict] = []

    for ds in targets:
        dirs = [d for d in ds["img_dirs"] if d.is_dir()]
        images: List[Path] = []
        for d in dirs:
            images.extend(list_images(d))
        if args.limit > 0:
            images = images[: args.limit]
        if not images:
            print(f"[{ds['id']}] tak ada gambar (dirs={[str(x) for x in ds['img_dirs']]}) — lewati.")
            summary["datasets"][ds["id"]] = {"status": "no_images", "n": 0,
                                             "display": ds["display"]}
            continue

        print(f"[{ds['id']}] {len(images)} gambar <- {[str(x) for x in dirs]}")
        rows: List[Dict] = []
        n_overlay = 0
        t0 = time.time()
        for i, img_path in enumerate(images, 1):
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  [{i}/{len(images)}] SKIP (tak terbaca): {img_path.name}",
                      file=sys.stderr)
                continue
            try:
                metrics, overlay = roundtrip_one(net, img, device, use_fp16,
                                                 args.tps_cap, args.inv_iters)
            except Exception as e:  # noqa: BLE001 — laporkan, lanjut gambar berikutnya
                print(f"  [{i}/{len(images)}] GAGAL {img_path.name}: {e}", file=sys.stderr)
                continue

            row = {"image": img_path.name, **metrics}
            rows.append(row)

            if n_overlay < args.overlays:
                ov = make_overlay(overlay, img_path.stem)
                out_png = OUT_DIR / f"{ds['id']}_overlay_{n_overlay:02d}_{img_path.stem}.png"
                cv2.imwrite(str(out_png), ov)
                n_overlay += 1

            if i % 20 == 0 or i == len(images):
                dt = time.time() - t0
                print(f"  [{i}/{len(images)}] ok={len(rows)} ({dt:.1f}s, {dt / i:.3f}s/img)")

        # CSV per-gambar.
        csv_path = OUT_DIR / f"{ds['id']}_metrics.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["image", "recon_mae", "recon_psnr", "hole_rate", "edge_loss_rate"])
            for r in rows:
                w.writerow([r["image"], f"{r['recon_mae']:.4f}", f"{r['recon_psnr']:.4f}",
                            f"{r['hole_rate']:.6f}", f"{r['edge_loss_rate']:.6f}"])

        agg = _aggregate(rows) if rows else {"n": 0}
        agg["status"] = "ok" if rows else "no_valid"
        agg["display"] = ds["display"]
        summary["datasets"][ds["id"]] = agg
        all_rows.extend(rows)
        if rows:
            print(f"[{ds['id']}] mae={agg['recon_mae_mean']:.3f} "
                  f"psnr={agg['recon_psnr_mean']:.2f}dB "
                  f"hole={agg['hole_rate_mean']*100:.2f}% "
                  f"edge={agg['edge_loss_rate_mean']*100:.2f}%  -> {csv_path.name}")

    # Agregat overall lintas semua dataset yang diproses.
    overall = _aggregate(all_rows) if all_rows else {"n": 0}
    overall["status"] = "ok" if all_rows else "no_valid"
    summary["overall"] = overall

    summary_path = OUT_DIR / "roundtrip_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[roundtrip] tulis {summary_path}")
    if all_rows:
        print(f"[roundtrip] OVERALL n={overall['n']} mae={overall['recon_mae_mean']:.3f} "
              f"psnr={overall['recon_psnr_mean']:.2f}dB hole={overall['hole_rate_mean']*100:.2f}% "
              f"edge={overall['edge_loss_rate_mean']*100:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
