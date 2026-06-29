"""
mowa_rectify_iterative.py — Rektifikasi fisheye MOWA secara ITERATIF (N pass).

Tujuan (skripsi): menguji saran pembimbing untuk menerapkan MOWA "bolak-balik" —
jalankan MOWA pada citra asli -> output-1, lalu umpankan output-1 kembali ke MOWA
-> output-2, dan seterusnya. Nama baku metode ini adalah ITERATIVE / RECURSIVE
RECTIFICATION (test-time iterative refinement): koreksi geometri disempurnakan
lewat beberapa lintasan berturut-turut.

CATATAN RISET PENTING (harap jujur di skripsi, ini ABLASI bukan jaminan menang):
  Metode iteratif yang "benar" (ESIR CVPR'19, DocScanner IJCV'25, RAFT ECCV'20)
  menyempurnakan sebuah FLOW FIELD lalu me-resample citra ASLI SEKALI di akhir.
  Di sini kita menempuh jalur NAIF yang disarankan pembimbing: me-render output
  MOWA lalu MENGUMPANKANNYA KEMBALI sebagai citra baru. Ini TIDAK teruji di
  literatur dan berisiko:
    1. BLUR menumpuk  — tiap pass = satu interpolasi bilinear lagi, ketajaman turun.
    2. KONTEN TEPI hilang — area yang ter-warp keluar frame jadi hitam permanen;
       ayam di pinggir bisa terpotong dan tak bisa kembali.
    3. OVER-CORRECTION — MOWA dilatih pada input yang terdistorsi SEKALI; output-nya
       sendiri berada di luar distribusi (off-distribution), sehingga pass ke-2+ bisa
       "mengoreksi" citra yang sebenarnya sudah lurus -> malah melengkung balik.
  Karena itu skrip ini MENGUKUR konvergensi (mean pixel displacement antar pass):
  bila displacement mengecil cepat, pass tambahan tak berguna. Empiris diharapkan
  <= 2 pass yang bermanfaat. Perlakukan hasilnya sebagai bukti ablasi, bukan klaim.

Perbedaan dengan mowa_rectify.py (base):
  - Base = 1 pass. Skrip ini = N pass (--iterations, default 2).
  - Input pass-1 = citra asli; input pass-k = OUTPUT pass-(k-1).
  - Bounding box GT dikomposisikan menembus SETIAP pass (warp berantai); box yang
    keluar frame di pass manapun dibuang (dihitung).
  - Manifest mencatat mean pixel displacement PER PASS (indikator konvergensi).

Kode inti (jaringan, flow, warp) DIIMPOR ULANG dari mowa_rectify.py — tidak
diimplementasi ulang.

Contoh pemakaian (dari root proyek, venv khusus MOWA):
  .venv-mowa/Scripts/python.exe src/mowa_rectify_iterative.py \
      --input data/images/val \
      --labels data/labels/val \
      --output data/rectified/pio_val_iter2 \
      --iterations 2 \
      --mowa-root vendor/MOWA \
      --checkpoint vendor/MOWA/checkpoint/mowa_pretrained.pth \
      --limit 20

Struktur output (cermin base):
  <output>/images/*.jpg              citra terkoreksi setelah N pass (resolusi asli)
  <output>/labels/*.txt              bbox YOLO hasil warp berantai N pass
  <output>/mowa_iter_manifest.json   ringkasan run + displacement per pass
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

# ROOT proyek (parents[1] dari src/). Dipakai untuk default path vendor/checkpoint.
ROOT = Path(__file__).resolve().parents[1]

# Pastikan folder src/ ada di sys.path agar `import mowa_rectify` jalan meski skrip
# dipanggil dengan path relatif dari root (mis. src/mowa_rectify_iterative.py).
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# Reuse SEMUA komponen MOWA dari base — jangan diimplementasi ulang.
from mowa_rectify import (  # noqa: E402
    INPUT_SIZE,
    add_mowa_to_path,
    build_net,
    compute_flows,
    list_images,
    load_checkpoint,
    read_yolo_labels,
    warp_boxes_via_flow,
    write_yolo_labels,
    _apply_full_warp,
)


@torch.no_grad()
def rectify_iterative_one(
    net,
    img_bgr: np.ndarray,
    boxes: Optional[np.ndarray],
    iterations: int,
    device: torch.device,
    tps_points: List[int],
    use_fp16: bool,
    tps_cap: int,
) -> Tuple[np.ndarray, Optional[List[Tuple[int, int, int, int, int]]], List[float], int, int]:
    """Luruskan satu gambar BGR secara iteratif N pass + komposisi bbox berantai.

    Return: (final_img_bgr, final_boxes_for_write | None, per_pass_disp, final_w, final_h)
      - final_boxes_for_write: list (orig_class_idx, x1,y1,x2,y2) pixel pada citra akhir,
        siap diberikan ke write_yolo_labels bersama `classes` asli. None bila tak ada label.
      - per_pass_disp: mean pixel displacement TIAP pass (lihat definisi di bawah).

    Alur: current = citra asli. Untuk pass 1..N:
      1. bangun input 256x256 dari `current`,
      2. compute_flows -> (tps2flow, flow) pada resolusi `current`,
      3. warp `current` via _apply_full_warp(bilinear) -> next_img,
      4. catat displacement pass ini,
      5. komposisikan bbox: warp box frame-`current` -> frame-`next_img`,
      6. current = next_img.
    Resolusi konstan antar pass (warp mempertahankan H,W), jadi displacement pixel
    langsung sebanding antar pass.

    DEFINISI displacement (didokumentasikan tegas):
      _apply_full_warp menerapkan DUA tahap resample backward: tps2flow lalu flow,
      keduanya field offset (1,2,H,W) dalam PIXEL resolusi asli (output pixel ->
      offset ke lokasi sumber). Peta-mundur gabungan pass ini kira-kira
      identitas + (tps2flow + flow) (aproksimasi orde-1; komposisi eksaknya
      mengevaluasi tps2flow pada identitas+flow, tapi untuk field TPS yang mulus dan
      flow residual kecil, penjumlahan langsung sudah representatif). Kita SENGAJA
      memakai penjumlahan field ini alih-alih me-warp peta koordinat identitas,
      untuk MENGHINDARI artefak tepi grid_sample (padding zeros pada _apply_full_warp
      akan meledakkan displacement di piksel yang memetakan ke luar frame).
        disp_field = tps2flow + flow          # (1,2,H,W), pixel
        mean_disp  = mean( sqrt(dx^2 + dy^2) ) # rata-rata magnitudo koreksi pass ini
      Konvergensi = mean_disp yang MENGECIL antar pass.
    """
    current = img_bgr
    per_pass_disp: List[float] = []

    # State bbox: cur_boxes (Nx4 xyxy pixel pada frame `current`), orig_idx memetakan
    # baris ke indeks box ASLI (untuk melacak class & drop menembus banyak pass).
    if boxes is not None and len(boxes) > 0:
        cur_boxes: Optional[np.ndarray] = np.asarray(boxes, dtype=np.float32)
        orig_idx: List[int] = list(range(len(cur_boxes)))
    else:
        cur_boxes = None
        orig_idx = []

    for _pass in range(iterations):
        h, w = current.shape[:2]

        input1 = np.transpose(current.astype(np.float32) / 255.0, (2, 0, 1))[None]  # 1,3,H,W
        resized = cv2.resize(current, (INPUT_SIZE, INPUT_SIZE)).astype(np.float32) / 255.0
        input2 = np.transpose(resized, (2, 0, 1))[None]  # 1,3,256,256
        mask = np.ones((1, 1, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)  # frame penuh

        input1_t = torch.from_numpy(input1).float().to(device)
        input2_t = torch.from_numpy(input2).float().to(device)
        mask_t = torch.from_numpy(mask).float().to(device)

        tps2flow, flow = compute_flows(
            net, input2_t, mask_t, h, w, tps_points, device, use_fp16, tps_cap=tps_cap
        )

        # Displacement pass ini (lihat DEFINISI di docstring).
        disp_field = tps2flow + flow  # 1,2,H,W (pixel)
        mag = torch.sqrt(disp_field[:, 0, :, :] ** 2 + disp_field[:, 1, :, :] ** 2)
        per_pass_disp.append(float(mag.mean().item()))

        # Warp citra (bilinear) -> input pass berikutnya.
        warp = _apply_full_warp(input1_t, tps2flow, flow, mode="bilinear")[0]
        warp_np = (warp.clamp(0, 1) * 255.0).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        if warp_np.shape[:2] != (h, w):
            warp_np = cv2.resize(warp_np, (w, h))

        # Komposisi bbox: warp box dari frame `current` -> frame `warp_np`.
        if cur_boxes is not None and len(cur_boxes) > 0:
            warped = warp_boxes_via_flow(cur_boxes, tps2flow, flow, h, w, device)
            new_boxes = np.zeros((len(warped), 4), dtype=np.float32)
            new_orig: List[int] = []
            for j, (row, x1, y1, x2, y2) in enumerate(warped):
                new_boxes[j] = (x1, y1, x2, y2)
                new_orig.append(orig_idx[row])
            cur_boxes = new_boxes
            orig_idx = new_orig

        current = warp_np

    final_h, final_w = current.shape[:2]
    final_write: Optional[List[Tuple[int, int, int, int, int]]] = None
    if boxes is not None:
        final_write = []
        if cur_boxes is not None:
            for j in range(len(cur_boxes)):
                x1, y1, x2, y2 = cur_boxes[j]
                final_write.append(
                    (orig_idx[j], int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
                )
    return current, final_write, per_pass_disp, final_w, final_h


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Rektifikasi fisheye MOWA ITERATIF (N pass) untuk YOLO dataset."
    )
    ap.add_argument("--input", required=True, type=Path, help="Folder gambar sumber (images/).")
    ap.add_argument("--labels", type=Path, default=None, help="Folder label YOLO (.txt) opsional.")
    ap.add_argument("--output", required=True, type=Path,
                    help="Folder output (dibuat: images/, labels/).")
    ap.add_argument("--iterations", type=int, default=2,
                    help="Jumlah pass MOWA berturut-turut (default 2). Pass-1=asli, pass-k=output pass-(k-1).")
    ap.add_argument("--mowa-root", type=Path, default=ROOT / "vendor" / "MOWA",
                    help="Root repo MOWA.")
    ap.add_argument("--checkpoint", type=Path,
                    default=ROOT / "vendor" / "MOWA" / "checkpoint" / "mowa_pretrained.pth")
    ap.add_argument("--limit", type=int, default=0, help="Proses N gambar pertama saja (0 = semua).")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--no-fp16", action="store_true", help="Nonaktifkan autocast fp16 (default fp16 ON).")
    ap.add_argument("--tps-cap", type=int, default=384,
                    help="Sisi terpanjang komputasi TPS coarse (default 384). 0 = full-res (lambat).")
    ap.add_argument("--ext", default=".jpg", help="Ekstensi file output gambar (default .jpg).")
    args = ap.parse_args()

    if args.iterations < 1:
        print(f"ERROR: --iterations harus >= 1 (diberi {args.iterations}).", file=sys.stderr)
        return 2
    if not args.input.is_dir():
        print(f"ERROR: --input bukan folder: {args.input}", file=sys.stderr)
        return 2

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        # resample_image_xy / jalur TPS MOWA meng-hardcode .cuda(); CPU akan gagal.
        print("ERROR: MOWA butuh CUDA (utils_transform meng-hardcode .cuda()). "
              "Tidak ada GPU terdeteksi.", file=sys.stderr)
        return 3

    add_mowa_to_path(args.mowa_root)

    tps_points = [10, 12, 14, 16]
    print(f"[mowa_iter] device={device} iterations={args.iterations}")
    net = build_net(device)
    load_checkpoint(net, args.checkpoint, device)
    net.eval()
    print(f"[mowa_iter] model dimuat dari {args.checkpoint}")

    do_labels = args.labels is not None
    use_fp16 = not args.no_fp16

    out_img_dir = args.output / "images"
    out_lbl_dir = args.output / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    if do_labels:
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(args.input)
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        print(f"ERROR: tidak ada gambar di {args.input}", file=sys.stderr)
        return 2

    ok, failed = 0, 0
    boxes_in_total, boxes_out_total = 0, 0
    fail_names: List[str] = []
    # Akumulasi displacement per indeks pass (dijumlah lalu dirata-ratakan atas ok).
    disp_sums = [0.0] * args.iterations
    t0 = time.time()

    for i, img_path in enumerate(images, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            failed += 1
            fail_names.append(img_path.name)
            print(f"  [{i}/{len(images)}] SKIP (tak terbaca): {img_path.name}", file=sys.stderr)
            continue

        ori_h, ori_w = img.shape[:2]
        classes: List[int] = []
        boxes = None
        if do_labels:
            src_lbl = args.labels / (img_path.stem + ".txt")
            classes, boxes = read_yolo_labels(src_lbl, ori_w, ori_h)

        try:
            final_img, final_write, per_pass_disp, final_w, final_h = rectify_iterative_one(
                net, img, boxes, args.iterations, device, tps_points,
                use_fp16=use_fp16, tps_cap=args.tps_cap,
            )
        except Exception as e:  # noqa: BLE001 — laporkan, lanjut gambar berikutnya
            failed += 1
            fail_names.append(img_path.name)
            print(f"  [{i}/{len(images)}] GAGAL rectify {img_path.name}: {e}", file=sys.stderr)
            continue

        out_name = img_path.stem + args.ext
        cv2.imwrite(str(out_img_dir / out_name), final_img)

        if do_labels:
            n_written = write_yolo_labels(
                out_lbl_dir / (img_path.stem + ".txt"), classes, final_write or [], final_w, final_h)
            boxes_in_total += len(classes)
            boxes_out_total += n_written

        # Akumulasi displacement (panjang == iterations untuk tiap gambar sukses).
        for k in range(args.iterations):
            disp_sums[k] += per_pass_disp[k]

        ok += 1
        if i % 20 == 0 or i == len(images):
            dt = time.time() - t0
            extra = f" boxes {boxes_out_total}/{boxes_in_total}" if do_labels else ""
            print(f"  [{i}/{len(images)}] ok={ok} failed={failed} "
                  f"({dt:.1f}s, {dt / i:.3f}s/img){extra}")

    per_pass_mean_disp = [round(disp_sums[k] / ok, 4) if ok else 0.0
                          for k in range(args.iterations)]

    manifest = {
        "input": str(args.input),
        "labels": str(args.labels) if args.labels else None,
        "output": str(args.output),
        "checkpoint": str(args.checkpoint),
        "model": "MOWA (TPAMI 2025, KangLiao929/MOWA), S-Lab License 1.0 non-commercial",
        "method": "iterative/recursive rectification (test-time iterative refinement, naive re-feed)",
        "input_size": INPUT_SIZE,
        "iterations": args.iterations,
        "label_mode": "warp",
        "fp16": use_fp16,
        "tps_cap": args.tps_cap,
        "total": len(images),
        "ok": ok,
        "failed": failed,
        "failed_names": fail_names,
        "per_pass_mean_displacement_px": per_pass_mean_disp,
        "displacement_definition": (
            "mean L2 dari (tps2flow + flow) per pass, dalam pixel resolusi asli; "
            "aproksimasi orde-1 offset peta-mundur gabungan. Menurun antar-pass = konvergen."
        ),
        "boxes_in": boxes_in_total,
        "boxes_out": boxes_out_total,
        "boxes_dropped": boxes_in_total - boxes_out_total,
        "seconds": round(time.time() - t0, 2),
        "sec_per_img": round((time.time() - t0) / max(1, ok + failed), 3),
        "device": str(device),
        "caveat": (
            "Re-feed NAIF (render output MOWA lalu diumpan balik) TIDAK teruji di literatur. "
            "Risiko: (1) blur menumpuk tiap pass akibat interpolasi bilinear berulang; "
            "(2) konten tepi hilang permanen (ayam di pinggir terpotong -> hitam, box dibuang); "
            "(3) over-correction karena MOWA dilatih pada distorsi SEKALI sehingga output-nya "
            "off-distribution dan pass ke-2+ bisa melengkungkan citra yang sudah lurus. "
            "Perlakukan sebagai ABLASI konvergensi (lihat per_pass_mean_displacement_px), "
            "bukan klaim kenaikan akurasi. Umumnya <= 2 pass yang berguna."
        ),
    }
    (args.output / "mowa_iter_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[mowa_iter] SELESAI: ok={ok} failed={failed} iterations={args.iterations} -> {args.output}")
    print(f"[mowa_iter] displacement per pass (px): {per_pass_mean_disp}")
    print(f"[mowa_iter] manifest: {args.output / 'mowa_iter_manifest.json'}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
