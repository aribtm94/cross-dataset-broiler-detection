"""
mowa_rectify.py — Batch fisheye/wide-angle rectification preprocessor using MOWA.

Tujuan (skripsi): mengganti heuristik "DaFIR-light" statistik di
`compare_camera_corrections.py` dengan model END-TO-END terlatih (MOWA, TPAMI 2025)
sebagai tahap preprocessing di depan YOLO11. Script ini membaca sebuah folder
dataset YOLO (images/ + labels/), meluruskan tiap gambar, dan menuliskan hasilnya
ke folder cermin yang siap dikonsumsi Ultralytics untuk eksperimen A/B.

Model: KangLiao929/MOWA (S-Lab License 1.0, non-komersial — aman untuk riset).
  - Input jaringan : citra 256x256 RGB [0,1] + 1 kanal mask (in_chans=4).
  - Task fisheye/wide-angle terdeteksi OTOMATIS oleh point-classifier internal;
    tidak perlu memberi task-id saat inference.
  - Output yang dipakai: `warp_flow` = citra terkoreksi pada resolusi ASLI.

CATATAN PENTING soal label:
  Rectification menggeser geometri gambar, sehingga bounding box asli tidak lagi
  presisi pada gambar hasil. Secara default script MENYALIN label apa adanya
  (--label-mode copy) supaya Ultralytics tetap bisa jalan untuk kondisi
  "rectify-test-only". Untuk perbandingan yang benar-benar adil, latih ulang /
  fine-tune YOLO11 pada gambar rectified (train + val sama-sama di-rectify), atau
  gunakan --label-mode warp (transformasi bbox via warping flow — TODO, belum
  diimplementasi; lihat argumen). Jangan mengklaim kenaikan akurasi dari
  "rectify-test-only" tanpa menyadari domain mismatch train/test.

Contoh pemakaian (dari root proyek, pakai venv khusus MOWA):
  .venv-mowa/Scripts/python.exe src/mowa_rectify.py \
      --input data/images/val \
      --labels data/labels/val \
      --output data/rectified/pio_val \
      --mowa-root vendor/MOWA \
      --checkpoint vendor/MOWA/checkpoint/mowa_pretrained.pth \
      --limit 20 --save-compare

Knob pelemah/coarse/jaga-tepi (default = warp penuh seperti dulu; lihat argparse):
  --coarse-only          pakai TPS coarse saja (buang residual flow padat)
  --warp-alpha 0.5       perlemah warp 50% (0=identitas, 1=penuh)
  --pad-frac 0.15        perbesar kanvas agar ayam/box tepi tak dibuang
  --seg-labels DIR       warp bbox via poligon mask ayam (Lever B), bukan persegi
  Contoh: ... --coarse-only --pad-frac 0.15 --output data/rectified_coarse/pio_val

Struktur output:
  <output>/images/*.jpg         citra terkoreksi (resolusi asli)
  <output>/labels/*.txt         label YOLO (disalin apa adanya secara default)
  <output>/compare/*.jpg        (opsional) sandingan asli|rectified untuk cek mata
  <output>/mowa_rectify_manifest.json  ringkasan run
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
INPUT_SIZE = 256
# Indeks task pada test_path default MOWA: 4 = fisheye, 1 = wide-angle. Hanya untuk
# dokumentasi — inference tidak butuh task-id (dideteksi otomatis point-classifier).
TASK_FISHEYE = 4


def add_mowa_to_path(mowa_root: Path) -> None:
    """MOWA memakai import absolut seperti `from model.network import MOWA`, jadi
    root repo MOWA harus ada di sys.path (bukan folder scripts kita)."""
    mowa_root = mowa_root.resolve()
    if not (mowa_root / "model" / "network.py").exists():
        raise FileNotFoundError(
            f"MOWA repo tidak ditemukan di {mowa_root}. "
            f"Clone dulu: git clone https://github.com/KangLiao929/MOWA vendor/MOWA"
        )
    if str(mowa_root) not in sys.path:
        sys.path.insert(0, str(mowa_root))


def build_net(device: torch.device):
    """Bangun jaringan MOWA dengan hyperparameter default dari test.py."""
    from model.network import MOWA

    net = MOWA(
        img_size=INPUT_SIZE,
        tps_points=[10, 12, 14, 16],
        embed_dim=32,
        win_size=8,
        token_projection="linear",
        token_mlp="leff",  # default; hindari dependensi torch_dwconv (fastleff)
        depths=[2, 2, 2, 2, 2, 2, 2, 2, 2],
        prompt=True,
        task_classes=6,
        head_num=4,
        shared_head=False,
    )
    return net.to(device)


def load_checkpoint(net, checkpoint_path: Path, device: torch.device) -> None:
    """Muat state_dict, tangani prefix 'module.' sama seperti test.py."""
    from collections import OrderedDict

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint tidak ditemukan: {checkpoint_path}\n"
            f"Unduh dari Google Drive (1fxQbD1TLoRnW8lG2a8KMinmD6Jlol8EX) atau Baidu, "
            f"lalu taruh di folder checkpoint MOWA."
        )
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    new_state = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state[name] = v
    net.load_state_dict(new_state)


def list_images(input_dir: Path) -> List[Path]:
    return sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


# ---------------------------------------------------------------------------
# Jalur inferensi TEROPTIMASI (menggantikan build_model_test vendor).
#
# Analisis bottleneck vendor (build_model_test + utils_transform):
#   1. Loop 4 head TPS menghitung resample full-res untuk SEMUA head, padahal
#      hanya head terakhir ([-1]) yang dipakai -> 3x kerja terberat dibuang.
#   2. get_coordinate_xy membangun grid identitas full-res dengan double-loop
#      Python (~2 juta iterasi) setiap panggilan resample_image_xy.
#
# Optimasi di sini:
#   - Forward jaringan (256x256) di bawah autocast fp16.
#   - Hitung HANYA head terakhir: tps2flow (TPS) + flow residual.
#   - Cache grid identitas full-res (torch, dibuat sekali per resolusi (H,W)).
#   - resample dua tahap yang SAMA dipakai ulang untuk gambar (bilinear) dan
#     untuk instance-mask label (nearest) -> lihat warp_boxes_via_flow.
# ---------------------------------------------------------------------------

_IDENTITY_GRID_CACHE: Dict[Tuple[int, int, str], torch.Tensor] = {}


def _identity_grid_xy(h: int, w: int, device: torch.device) -> torch.Tensor:
    """Grid koordinat identitas (1,2,H,W): channel 0 = x (kolom), 1 = y (baris).

    Pengganti torch-native untuk get_coordinate_xy vendor yang memakai double-loop
    Python. Di-cache per (H,W,device) karena semua gambar satu dataset seragam.
    """
    key = (h, w, str(device))
    g = _IDENTITY_GRID_CACHE.get(key)
    if g is None:
        ys, xs = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing="ij",
        )
        g = torch.stack([xs, ys], dim=0).unsqueeze(0)  # 1,2,H,W
        _IDENTITY_GRID_CACHE[key] = g
    return g


def _resample_xy(feature: torch.Tensor, flow: torch.Tensor, mode: str = "bilinear",
                 out_hw: Optional[Tuple[int, int]] = None,
                 offset: Tuple[int, int] = (0, 0)) -> torch.Tensor:
    """Setara resample_image_xy vendor tapi grid identitas di-cache & mode bebas.

    flow adalah BACKWARD map: untuk tiap pixel OUTPUT, uv = identitas + flow menunjuk
    posisi sampel di INPUT. Dinormalisasi ke [-1,1] lalu grid_sample.

    out_hw/offset OPSIONAL (untuk padding kanvas, lihat _resample_padded):
      - Bila out_hw is None dan offset==(0,0) -> jalur LAMA persis (nol drift).
      - Bila diisi: grid output berukuran out_hw, koordinat identitasnya digeser
        `-offset` (piksel output (X,Y) mewakili posisi logis (X-ox, Y-oy)), lalu
        flow ditambahkan. Normalisasi TETAP memakai dimensi INPUT (w-1)/2,(h-1)/2 —
        agar kanvas yang diperbesar bisa menyampel piksel input; koordinat di luar
        input jatuh ke padding_mode='zeros' (margin hitam yang memang diinginkan).
    """
    b, _, h, w = feature.shape
    x0 = (w - 1) / 2.0
    y0 = (h - 1) / 2.0
    if out_hw is None and offset == (0, 0):
        uv = _identity_grid_xy(h, w, feature.device) + flow  # 1,2,H,W (jalur lama)
    else:
        oh, ow = out_hw if out_hw is not None else (h, w)
        ident = _identity_grid_xy(oh, ow, feature.device).clone()  # 1,2,oh,ow
        ident[:, 0, :, :] -= offset[0]
        ident[:, 1, :, :] -= offset[1]
        uv = ident + flow  # flow harus (1,2,oh,ow)
    nx = (uv[:, 0, :, :] - x0) / x0
    ny = (uv[:, 1, :, :] - y0) / y0
    grid = torch.stack([nx, ny], dim=-1)  # 1,oh,ow,2
    if grid.shape[0] != b:
        grid = grid.expand(b, -1, -1, -1)
    return F.grid_sample(feature, grid, mode=mode, align_corners=True)


def _coarse_size(ori_h: int, ori_w: int, cap: int) -> Tuple[int, int]:
    """Ukuran coarse untuk komputasi TPS (sisi terpanjang <= cap). TPS mulus,
    jadi hitung di resolusi kecil lalu upsample nyaris tanpa galat geometrik."""
    if cap <= 0 or max(ori_h, ori_w) <= cap:
        return ori_h, ori_w
    s = cap / float(max(ori_h, ori_w))
    return max(1, int(round(ori_h * s))), max(1, int(round(ori_w * s)))


def _upscale_flow(flow_c: torch.Tensor, ori_h: int, ori_w: int) -> torch.Tensor:
    """Upsample field flow (1,2,ch,cw) -> (1,2,H,W) dan skala offset pixel-nya."""
    _, _, ch, cw = flow_c.shape
    if (ch, cw) == (ori_h, ori_w):
        return flow_c
    flow = F.interpolate(flow_c, size=(ori_h, ori_w), mode="bilinear", align_corners=True)
    flow[:, 0, :, :] *= ori_w / cw
    flow[:, 1, :, :] *= ori_h / ch
    return flow


@torch.no_grad()
def compute_flows(net, input2_t: torch.Tensor, mask_t: torch.Tensor, ori_h: int, ori_w: int,
                  tps_points: List[int], device: torch.device, use_fp16: bool,
                  tps_cap: int = 384):
    """Forward jaringan + bangun tps2flow (head terakhir) & flow residual full-res.

    Mengembalikan (tps2flow, flow) keduanya (1,2,H,W) pada resolusi asli. Keduanya
    dipakai ulang untuk gambar maupun instance-mask agar warp konsisten.

    OPTIMASI KUNCI: TPS transformer vendor membangun matriks [1, npoint+3, H*W]
    (~2 GB pada 1920x1080) dalam fp32 -> ini bottleneck sebenarnya (bukan jumlah
    head / bukan grid loop). Karena field TPS mulus secara spasial, kita hitung di
    resolusi coarse (sisi terpanjang <= tps_cap) lalu upsample+skala. Galat
    geometrik dapat diabaikan; kecepatan naik ~ (H*W)/(ch*cw) kali.
    """
    import utils.torch_tps_upsample as torch_tps_upsample
    from utils.utils_transform import get_rigid_mesh, get_norm_mesh

    autocast = torch.cuda.amp.autocast if (use_fp16 and device.type == "cuda") else None
    if autocast is not None:
        with autocast():
            offset, flow, _point_cls = net(input2_t, mask_t)
    else:
        offset, flow, _point_cls = net(input2_t, mask_t)

    # Hanya head terakhir (kontrak build_model_test memakai output_tps_list[-1] & flow).
    last = len(offset) - 1
    n_pts = tps_points[last]
    mesh_motion = offset[last].reshape(-1, n_pts, n_pts, 2).float()
    rigid_mesh = get_rigid_mesh(1, INPUT_SIZE, INPUT_SIZE, n_pts - 1, n_pts - 1)
    ori_mesh = rigid_mesh + mesh_motion
    clamped_x = torch.clamp(ori_mesh[..., 0], min=0, max=INPUT_SIZE - 1)
    clamped_y = torch.clamp(ori_mesh[..., 1], min=0, max=INPUT_SIZE - 1)
    ori_mesh = torch.stack((clamped_x, clamped_y), dim=-1)

    norm_rigid_mesh = get_norm_mesh(rigid_mesh, INPUT_SIZE, INPUT_SIZE)
    norm_ori_mesh = get_norm_mesh(ori_mesh, INPUT_SIZE, INPUT_SIZE)

    ch, cw = _coarse_size(ori_h, ori_w, tps_cap)
    tps2flow_c = torch_tps_upsample.transformer(norm_rigid_mesh, norm_ori_mesh, (ch, cw))
    tps2flow = _upscale_flow(tps2flow_c, ori_h, ori_w)

    # flow residual di-upsample ke resolusi asli (branch resize_flow=True vendor).
    flow = flow.float()
    flow = F.interpolate(flow, size=(ori_h, ori_w), mode="bilinear", align_corners=True)
    scale_h, scale_w = ori_h / INPUT_SIZE, ori_w / INPUT_SIZE
    flow[:, 0, :, :] *= scale_w
    flow[:, 1, :, :] *= scale_h
    return tps2flow, flow


def _apply_full_warp(feature: torch.Tensor, tps2flow: torch.Tensor,
                     flow: Optional[torch.Tensor], mode: str) -> torch.Tensor:
    """Terapkan dua tahap warp yang sama seperti vendor:
       output_tps = resample(feature, tps2flow); output = resample(output_tps, flow).

    Bila flow is None -> kembalikan output_tps saja (mode COARSE-only / TPS-only,
    membuang residual flow padat yang menjadi sumber distorsi lokal box). Semua
    pemanggil lama mengirim `flow` nyata sehingga perilakunya tak berubah.
    """
    output_tps = _resample_xy(feature, tps2flow, mode=mode)
    if flow is None:
        return output_tps
    output = _resample_xy(output_tps, flow, mode=mode)
    return output


def _combined_disp(tps2flow: torch.Tensor, flow: Optional[torch.Tensor],
                   h: int, w: int, device: torch.device) -> torch.Tensor:
    """Peta perpindahan GABUNGAN (backward) dari kedua tahap warp: Dcomb = M - identitas,
    dengan M = _apply_full_warp(identitas). Untuk tiap piksel output p, M(p) adalah
    koordinat sumber di INPUT (identik dengan compute_inverse_map di roundtrip_bbox_remap).
    Return (1,2,H,W).
    """
    ident = _identity_grid_xy(h, w, device)
    m = _apply_full_warp(ident, tps2flow, flow, mode="bilinear")  # 1,2,H,W koord sumber
    return m - ident


def _resample_padded(feature: torch.Tensor, dcomb: torch.Tensor,
                     ox: int, oy: int, hp: int, wp: int, mode: str) -> torch.Tensor:
    """Sampel `feature` (H×W) ke kanvas diperbesar Hp×Wp memakai peta gabungan `dcomb`.

    dcomb (1,2,H,W) di-pad 'replicate' sebesar (ox,oy) menjadi (1,2,Hp,Wp) sehingga
    margin baru mempertahankan perpindahan tepi (edge chickens tetap tersampel, bukan
    dibuang). Lalu satu kali _resample_xy dgn out_hw=(Hp,Wp), offset=(ox,oy).

    CATATAN: jalur ini menerapkan SATU interpolasi (bukan dua tahap seperti
    _apply_full_warp); untuk piksel interior hasilnya setara M(p) karena dcomb sudah
    memuat komposisi kedua tahap. Sedikit smoothing ekstra hanya di margin — dapat
    diabaikan; jalur pad=0 tetap memakai _apply_full_warp dua-tahap yang eksak.
    """
    dpad = F.pad(dcomb, (ox, ox, oy, oy), mode="replicate")  # 1,2,Hp,Wp
    return _resample_xy(feature, dpad, mode=mode, out_hw=(hp, wp), offset=(ox, oy))


@torch.no_grad()
def rectify_one(net, build_model_test, img_bgr: np.ndarray, device: torch.device,
                tps_points: List[int], use_fp16: bool = True, legacy: bool = False,
                boxes: Optional[np.ndarray] = None, tps_cap: int = 384,
                warp_alpha: float = 1.0, coarse_only: bool = False,
                pad_frac: float = 0.0,
                masks: Optional[List[Optional[np.ndarray]]] = None):
    """Luruskan satu gambar BGR (uint8, resolusi asli) -> BGR uint8 rectified.

    Jika `boxes` diberikan (Nx4 xyxy pixel), sekaligus mengembalikan bbox hasil warp
    lewat instance-mask (lihat warp_boxes_via_flow). Return: (warp_np, warped_boxes|None).

    Knob (default = perilaku lama persis):
      warp_alpha  : skala kedua field perpindahan (1.0=penuh, 0=identitas) — memperlemah warp.
      coarse_only : True -> pakai TPS coarse saja, buang residual flow padat.
      pad_frac    : >0 -> perbesar kanvas keluaran (1+2p) agar konten/box tepi tak dibuang.
      masks       : list poligon per-box (Nx2 piksel) untuk warp_boxes_via_flow (Lever B);
                    None -> fallback rasterisasi persegi seperti sebelumnya.

    legacy=True memakai build_model_test vendor (untuk sanity-check numerik).
    """
    ori_h, ori_w = img_bgr.shape[:2]

    input1 = img_bgr.astype(np.float32) / 255.0
    input1 = np.transpose(input1, (2, 0, 1))[None]  # 1,3,H,W
    resized = cv2.resize(img_bgr, (INPUT_SIZE, INPUT_SIZE)).astype(np.float32) / 255.0
    input2 = np.transpose(resized, (2, 0, 1))[None]  # 1,3,256,256
    mask = np.ones((1, 1, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)  # frame penuh

    input1_t = torch.from_numpy(input1).float().to(device)
    input2_t = torch.from_numpy(input2).float().to(device)
    mask_t = torch.from_numpy(mask).float().to(device)

    if legacy:
        out = build_model_test(net, input1_t, input2_t, mask_t, tps_points, resize_flow=True)
        warp = out["warp_flow"][0]
        warp_np = (warp.clamp(0, 1) * 255.0).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        if warp_np.shape[:2] != (ori_h, ori_w):
            warp_np = cv2.resize(warp_np, (ori_w, ori_h))
        return warp_np, None

    tps2flow, flow = compute_flows(net, input2_t, mask_t, ori_h, ori_w, tps_points, device,
                                   use_fp16, tps_cap=tps_cap)

    # Lever A: perlemah lalu (opsional) buang residual flow -> warp lebih coarse/mulus.
    if warp_alpha != 1.0:
        tps2flow = tps2flow * warp_alpha
        if flow is not None:
            flow = flow * warp_alpha
    if coarse_only:
        flow = None

    # Lever "jaga tepi": kanvas diperbesar agar edge chickens tak keluar frame & dibuang.
    if pad_frac and pad_frac > 0:
        ox = int(round(pad_frac * ori_w))
        oy = int(round(pad_frac * ori_h))
        hp, wp = ori_h + 2 * oy, ori_w + 2 * ox
        dcomb = _combined_disp(tps2flow, flow, ori_h, ori_w, device)
        warp = _resample_padded(input1_t, dcomb, ox, oy, hp, wp, mode="bilinear")[0]
        target_h, target_w = hp, wp
    else:
        ox = oy = 0
        hp, wp = ori_h, ori_w
        dcomb = None
        warp = _apply_full_warp(input1_t, tps2flow, flow, mode="bilinear")[0]
        target_h, target_w = ori_h, ori_w

    warp_np = (warp.clamp(0, 1) * 255.0).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
    if warp_np.shape[:2] != (target_h, target_w):
        warp_np = cv2.resize(warp_np, (target_w, target_h))

    warped_boxes = None
    if boxes is not None:
        warped_boxes = warp_boxes_via_flow(
            boxes, tps2flow, flow, ori_h, ori_w, device,
            masks=masks, dcomb=dcomb, pad=(ox, oy), out_hw=(hp, wp),
        )
    return warp_np, warped_boxes


def warp_boxes_via_flow(boxes: np.ndarray, tps2flow: torch.Tensor,
                        flow: Optional[torch.Tensor],
                        ori_h: int, ori_w: int, device: torch.device,
                        masks: Optional[List[Optional[np.ndarray]]] = None,
                        dcomb: Optional[torch.Tensor] = None,
                        pad: Tuple[int, int] = (0, 0),
                        out_hw: Optional[Tuple[int, int]] = None,
                        ) -> List[Tuple[int, int, int, int, int]]:
    """Warp bbox lewat instance-mask + flow yang SAMA dengan gambar.

    boxes: Nx4 (x1,y1,x2,y2) pixel pada gambar ASLI (input).
    Metode: lukis tiap objek ber-ID unik (i+1) pada peta single-channel resolusi asli,
    lalu resample NEAREST memakai warp yang sama seperti gambar. Untuk tiap ID yang
    masih muncul di output, ambil min/max x,y -> bbox rectified. ID yang hilang
    (ter-warp keluar frame) dibuang.

    Parameter opsional (default = perilaku lama persis):
      masks : list poligon per-box (Nx2 piksel gambar asli). Bila masks[i] valid
              (>=3 titik), objek dilukis via cv2.fillPoly (extent ketat mengikuti
              bentuk ayam) alih-alih persegi penuh -> menghilangkan pelebaran akibat
              rasterisasi persegi. masks[i] None/invalid -> fallback persegi.
      dcomb/pad/out_hw : bila pad != (0,0), warp id_map ke kanvas diperbesar (out_hw)
              via _resample_padded memakai peta gabungan dcomb (harus diberikan).
              Default pad=(0,0) -> jalur _apply_full_warp dua-tahap eksak seperti dulu.

    Return list (id_index, x1,y1,x2,y2) pixel pada gambar rectified (kanvas out_hw
    bila padding aktif).
    """
    n = len(boxes)
    if n == 0:
        return []
    # Peta ID: 0 = background, i+1 = box ke-i. float32 grid_sample nearest aman sampai
    # ribuan id (broiler padat < 300/gambar).
    id_map = np.zeros((ori_h, ori_w), dtype=np.float32)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        poly = masks[i] if (masks is not None and i < len(masks)) else None
        if poly is not None and len(poly) >= 3:
            pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
            pts[:, 0] = np.clip(pts[:, 0], 0, ori_w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, ori_h - 1)
            cv2.fillPoly(id_map, [pts.round().astype(np.int32)], float(i + 1))
            continue
        xi1 = max(0, min(ori_w - 1, int(round(x1))))
        yi1 = max(0, min(ori_h - 1, int(round(y1))))
        xi2 = max(0, min(ori_w, int(round(x2))))
        yi2 = max(0, min(ori_h, int(round(y2))))
        if xi2 <= xi1 or yi2 <= yi1:
            continue
        id_map[yi1:yi2, xi1:xi2] = float(i + 1)

    id_t = torch.from_numpy(id_map)[None, None].to(device)  # 1,1,H,W
    if pad != (0, 0) and dcomb is not None:
        ox, oy = pad
        hp, wp = out_hw if out_hw is not None else (ori_h + 2 * oy, ori_w + 2 * ox)
        warped = _resample_padded(id_t, dcomb, ox, oy, hp, wp, mode="nearest")[0, 0]
    else:
        warped = _apply_full_warp(id_t, tps2flow, flow, mode="nearest")[0, 0]
    warped_np = warped.round().cpu().numpy().astype(np.int32)

    out: List[Tuple[int, int, int, int, int]] = []
    for i in range(n):
        ys, xs = np.where(warped_np == (i + 1))
        if xs.size == 0:
            continue  # box ter-warp keluar frame / tertutup total
        out.append((i, int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    return out


def read_yolo_labels(path: Path, img_w: int, img_h: int):
    """Baca .txt YOLO -> (classes: List[int], boxes_xyxy: np.ndarray Nx4 pixel).

    Format: class cx cy w h (ternormalisasi). Baris tak valid dilewati.
    """
    classes: List[int] = []
    rows: List[Tuple[float, float, float, float]] = []
    if not path.exists():
        return classes, np.zeros((0, 4), dtype=np.float32)
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        except ValueError:
            continue
        x1 = (cx - w / 2.0) * img_w
        y1 = (cy - h / 2.0) * img_h
        x2 = (cx + w / 2.0) * img_w
        y2 = (cy + h / 2.0) * img_h
        classes.append(cls)
        rows.append((x1, y1, x2, y2))
    return classes, np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, 4), dtype=np.float32)


def read_yolo_polygons(path: Path, img_w: int, img_h: int):
    """Baca .txt YOLO-seg -> (classes: List[int], polygons: List[np.ndarray (K,2) piksel]).

    Format tiap baris: class x1 y1 x2 y2 ... xk yk (ternormalisasi 0-1, >=3 titik).
    Baris dengan tepat 4 koordinat (2 titik) diperlakukan sebagai persegi 2-sudut ->
    dikembangkan ke 4 sudut, agar file campuran bbox/seg tetap terbaca. Baris tak
    valid (koordinat ganjil / < 2 titik) dilewati dengan poligon None pada indeksnya
    TIDAK — poligon hanya ditambahkan untuk baris valid; penyelarasan ke box dilakukan
    di pemanggil. Return classes & polygons sejajar (satu entri per baris valid).
    """
    classes: List[int] = []
    polygons: List[np.ndarray] = []
    if not path.exists():
        return classes, polygons
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        if len(coords) % 2 != 0 or len(coords) < 4:
            continue
        pts = np.asarray(coords, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] == 2:
            # 2 titik -> persegi (x1,y1)-(x2,y2) diperluas ke 4 sudut.
            (x1, y1), (x2, y2) = pts[0], pts[1]
            pts = np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        pts[:, 0] *= img_w
        pts[:, 1] *= img_h
        classes.append(cls)
        polygons.append(pts)
    return classes, polygons


def write_yolo_labels(path: Path, classes: List[int],
                      warped: List[Tuple[int, int, int, int, int]], img_w: int, img_h: int) -> int:
    """Tulis bbox hasil warp ke .txt YOLO (ternormalisasi). Return jumlah baris ditulis."""
    lines = []
    for (idx, x1, y1, x2, y2) in warped:
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h
        cx = (x1 + x2) / 2.0 / img_w
        cy = (y1 + y2) / 2.0 / img_h
        if bw <= 0 or bh <= 0:
            continue
        cls = classes[idx] if 0 <= idx < len(classes) else 0
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch MOWA fisheye rectification untuk YOLO dataset.")
    ap.add_argument("--input", required=True, type=Path, help="Folder gambar sumber (images/).")
    ap.add_argument("--labels", type=Path, default=None, help="Folder label YOLO (.txt) opsional.")
    ap.add_argument("--output", required=True, type=Path, help="Folder output (dibuat: images/, labels/).")
    ap.add_argument("--mowa-root", type=Path, default=Path("vendor/MOWA"), help="Root repo MOWA.")
    ap.add_argument("--checkpoint", type=Path, default=Path("vendor/MOWA/checkpoint/mowa_pretrained.pth"))
    ap.add_argument("--limit", type=int, default=0, help="Proses N gambar pertama saja (0 = semua).")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--label-mode", choices=["copy", "warp", "none"], default="warp",
                    help="warp: transformasi bbox via flow MOWA (instance-mask + nearest, DEFAULT). "
                         "copy: salin .txt apa adanya. none: jangan tulis label.")
    ap.add_argument("--save-compare", action="store_true",
                    help="Tulis sandingan asli|rectified ke <output>/compare untuk cek mata.")
    ap.add_argument("--save-label-preview", action="store_true",
                    help="Tulis overlay bbox hasil warp pada gambar rectified ke <output>/label_preview.")
    ap.add_argument("--no-fp16", action="store_true", help="Nonaktifkan autocast fp16 (default fp16 ON).")
    ap.add_argument("--tps-cap", type=int, default=384,
                    help="Sisi terpanjang komputasi TPS coarse (default 384). Naikkan untuk presisi, "
                         "turunkan untuk kecepatan. 0 = full-res (lambat).")
    ap.add_argument("--legacy", action="store_true",
                    help="Pakai build_model_test vendor (4 head, tanpa optimasi) untuk sanity-check.")
    ap.add_argument("--ext", default=".jpg", help="Ekstensi file output gambar (default .jpg).")
    # --- Knob Lever A (perlemah/coarse/jaga-tepi) & Lever B (mask). Default = perilaku lama. ---
    ap.add_argument("--coarse-only", action="store_true",
                    help="Pakai TPS coarse saja (buang residual flow padat). Warp lebih mulus, "
                         "distorsi lokal box berkurang.")
    ap.add_argument("--warp-alpha", type=float, default=1.0,
                    help="Skala kedua field perpindahan (1.0=penuh MOWA, 0=identitas). "
                         "<1 memperlemah warp -> box tak melebar/terpotong sebanyak default.")
    ap.add_argument("--pad-frac", type=float, default=0.0,
                    help="Perbesar kanvas keluaran sebesar frac di tiap sisi (mis. 0.15) agar "
                         "ayam/box di tepi tak ter-warp keluar frame lalu dibuang. 0 = tanpa padding.")
    ap.add_argument("--seg-labels", type=Path, default=None,
                    help="Folder label YOLO-seg (poligon) untuk warp bbox via mask ayam asli "
                         "(Lever B), bukan rasterisasi persegi. Diselaraskan per-indeks ke --labels.")
    args = ap.parse_args()

    if args.warp_alpha < 0:
        print(f"ERROR: --warp-alpha harus >= 0 (diberi {args.warp_alpha})", file=sys.stderr)
        return 2
    if args.pad_frac < 0:
        print(f"ERROR: --pad-frac harus >= 0 (diberi {args.pad_frac})", file=sys.stderr)
        return 2

    if not args.input.is_dir():
        print(f"ERROR: --input bukan folder: {args.input}", file=sys.stderr)
        return 2

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        # resample_image_xy di MOWA meng-hardcode .cuda(); CPU akan gagal.
        print("ERROR: MOWA butuh CUDA (utils_transform.resample_image_xy hardcode .cuda()). "
              "Tidak ada GPU terdeteksi.", file=sys.stderr)
        return 3

    add_mowa_to_path(args.mowa_root)
    from model.builder import build_model_test  # noqa: E402  (perlu sys.path dulu)

    tps_points = [10, 12, 14, 16]
    print(f"[mowa_rectify] device={device}")
    net = build_net(device)
    load_checkpoint(net, args.checkpoint, device)
    net.eval()
    print(f"[mowa_rectify] model dimuat dari {args.checkpoint}")

    do_labels = args.label_mode != "none" and args.labels is not None
    do_warp = args.label_mode == "warp" and args.labels is not None
    use_fp16 = not args.no_fp16

    out_img_dir = args.output / "images"
    out_lbl_dir = args.output / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    if do_labels:
        out_lbl_dir.mkdir(parents=True, exist_ok=True)
    compare_dir = args.output / "compare"
    if args.save_compare:
        compare_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = args.output / "label_preview"
    if args.save_label_preview:
        preview_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(args.input)
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        print(f"ERROR: tidak ada gambar di {args.input}", file=sys.stderr)
        return 2

    ok, failed = 0, 0
    boxes_in_total, boxes_out_total = 0, 0
    seg_used, seg_mismatch = 0, 0
    fail_names: List[str] = []
    t0 = time.time()
    for i, img_path in enumerate(images, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            failed += 1
            fail_names.append(img_path.name)
            print(f"  [{i}/{len(images)}] SKIP (tak terbaca): {img_path.name}", file=sys.stderr)
            continue

        ori_h, ori_w = img.shape[:2]
        src_lbl = (args.labels / (img_path.stem + ".txt")) if args.labels is not None else None
        classes: List[int] = []
        boxes = None
        masks = None
        if do_warp:
            classes, boxes = read_yolo_labels(src_lbl, ori_w, ori_h)
            # Lever B: muat poligon mask (bila --seg-labels) & selaraskan per-indeks ke boxes.
            if args.seg_labels is not None and len(classes) > 0:
                seg_path = args.seg_labels / (img_path.stem + ".txt")
                _seg_cls, seg_polys = read_yolo_polygons(seg_path, ori_w, ori_h)
                if len(seg_polys) == len(classes):
                    masks = seg_polys
                    seg_used += 1
                else:
                    # Jumlah tak cocok -> tak bisa dipetakan per-indeks; fallback persegi.
                    seg_mismatch += 1
                    if seg_mismatch <= 5:
                        print(f"  [seg] {img_path.stem}: poligon={len(seg_polys)} != "
                              f"box={len(classes)} -> fallback persegi", file=sys.stderr)

        try:
            rect, warped_boxes = rectify_one(
                net, build_model_test, img, device, tps_points,
                use_fp16=use_fp16, legacy=args.legacy, boxes=boxes, tps_cap=args.tps_cap,
                warp_alpha=args.warp_alpha, coarse_only=args.coarse_only,
                pad_frac=args.pad_frac, masks=masks,
            )
        except Exception as e:  # noqa: BLE001 — laporkan, lanjut gambar berikutnya
            failed += 1
            fail_names.append(img_path.name)
            print(f"  [{i}/{len(images)}] GAGAL rectify {img_path.name}: {e}", file=sys.stderr)
            continue

        out_name = img_path.stem + args.ext
        cv2.imwrite(str(out_img_dir / out_name), rect)
        # Normalisasi label memakai ukuran rect FINAL (penting saat --pad-frac memperbesar
        # kanvas). Saat pad=0, rw,rh == ori_w,ori_h -> tak ada perubahan perilaku.
        rh, rw = rect.shape[:2]

        if args.label_mode == "copy" and src_lbl is not None and src_lbl.exists():
            shutil.copy2(src_lbl, out_lbl_dir / (img_path.stem + ".txt"))
        elif do_warp:
            n_written = write_yolo_labels(
                out_lbl_dir / (img_path.stem + ".txt"), classes, warped_boxes or [], rw, rh)
            boxes_in_total += len(classes)
            boxes_out_total += n_written
            if args.save_label_preview and warped_boxes:
                prev = rect.copy()
                for (_idx, x1, y1, x2, y2) in warped_boxes:
                    cv2.rectangle(prev, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imwrite(str(preview_dir / (img_path.stem + ".jpg")), prev)

        if args.save_compare:
            h = min(img.shape[0], rect.shape[0])
            left = cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))
            right = cv2.resize(rect, (int(rect.shape[1] * h / rect.shape[0]), h))
            sep = np.full((h, 4, 3), 255, np.uint8)
            cv2.imwrite(str(compare_dir / (img_path.stem + "_cmp.jpg")),
                        np.hstack([left, sep, right]))

        ok += 1
        if i % 20 == 0 or i == len(images):
            dt = time.time() - t0
            extra = f" boxes {boxes_out_total}/{boxes_in_total}" if do_warp else ""
            print(f"  [{i}/{len(images)}] ok={ok} failed={failed} "
                  f"({dt:.1f}s, {dt / i:.3f}s/img){extra}")

    manifest = {
        "input": str(args.input),
        "labels": str(args.labels) if args.labels else None,
        "output": str(args.output),
        "checkpoint": str(args.checkpoint),
        "model": "MOWA (TPAMI 2025, KangLiao929/MOWA), S-Lab License 1.0 non-commercial",
        "input_size": INPUT_SIZE,
        "task_note": "fisheye/wide-angle auto-detected by MOWA point-classifier (no task-id at inference)",
        "label_mode": args.label_mode,
        "fp16": use_fp16,
        "legacy": args.legacy,
        "coarse_only": args.coarse_only,
        "warp_alpha": args.warp_alpha,
        "pad_frac": args.pad_frac,
        "seg_labels": str(args.seg_labels) if args.seg_labels else None,
        "seg_images_used": seg_used,
        "seg_mismatch_images": seg_mismatch,
        "total": len(images),
        "ok": ok,
        "failed": failed,
        "failed_names": fail_names,
        "boxes_in": boxes_in_total,
        "boxes_out": boxes_out_total,
        "boxes_dropped": boxes_in_total - boxes_out_total,
        "seconds": round(time.time() - t0, 2),
        "sec_per_img": round((time.time() - t0) / max(1, ok + failed), 3),
        "device": str(device),
        "caveat": (
            "label-mode=warp: bbox ditransformasi via flow MOWA (instance-mask + nearest) "
            "sehingga selaras dengan gambar rectified; box yang ter-warp keluar frame dibuang. "
            "label-mode=copy: label lama TIDAK selaras dengan geometri rectified. "
            "coarse_only/warp_alpha/pad_frac default (False/1.0/0.0) = perilaku warp penuh lama. "
            "pad_frac>0: kanvas keluaran diperbesar & label dinormalisasi ke ukuran itu; jalur pad "
            "memakai satu interpolasi gabungan (bukan dua tahap) — smoothing margin dapat diabaikan. "
            "seg_labels: bbox dilukis via poligon SAM (bukan persegi) untuk extent lebih ketat."
        ),
    }
    (args.output / "mowa_rectify_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[mowa_rectify] SELESAI: ok={ok} failed={failed} -> {args.output}")
    print(f"[mowa_rectify] manifest: {args.output / 'mowa_rectify_manifest.json'}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
