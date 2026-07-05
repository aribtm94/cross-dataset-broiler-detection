r"""
Train YOLOv8m di PIO — resep Tabel 7 tier-m, LANGSUNG yolov8m saja.
Turunan dari PIO_Model_Comparison_Colab.ipynb (yang menghasilkan cmp_yolo11m).

Fitur:
  1) Augmentasi OKLUSI (CoarseDropout lubang-kecil via albumentations) -> Run B (§rencana 13.1).
  2) Tambah dataset RILIS RECTIFIED PIO ke split TRAIN (val tetap PIO asli).
  3) Batch besar untuk RTX 5090, epoch tetap 100.

CARA JALAN DI PC RTX 5090 (project di E:\CCTV\PIO):
    cd E:\CCTV\PIO
    python train_yolov8m_pio.py
  (albumentations akan di-install otomatis bila belum ada.)

Toggle penting ada di blok KONFIGURASI di bawah:
  - USE_OCCLUSION = True  -> Run B (+oklusi).  False -> Run A (baseline).
  - RECTIFIED_PIO_DIR     -> SESUAIKAN path-nya (tetap di E:\CCTV\PIO). None utk nonaktif.
  - BATCH                 -> default 16 (muat 5090). A & B WAJIB pakai batch sama.

Nama run otomatis: cmp_yolov8m[_occ][_rect]  -> tidak saling menimpa, cocok utk ablation.
Output: runs_compare/<run>/weights/best.pt  + cetak P/R/mAP in-domain.

Lingkungan terverifikasi di PC 5090: ultralytics 8.3.152 · torch 2.11.0+cu128 · Python 3.11.
"""

from pathlib import Path
import os, sys, random, glob, re, collections, shutil, subprocess
import numpy as np

# ----------------------------------------------------------------------
# Reproducibility (seed 0, deterministic) — sama seperti notebook
# ----------------------------------------------------------------------
SEED = 0
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"   # matikan cek-update online albumentations
random.seed(SEED)
np.random.seed(SEED)

import torch
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

import yaml
import ultralytics
from ultralytics import YOLO

# ======================================================================
# KONFIGURASI (sesuaikan di sini)
# ======================================================================
BASE = Path(__file__).resolve().parent          # E:\CCTV\PIO di PC 5090
DATA_ROOT = BASE / "downloads" / "data"          # PIO mentah
RUNS_DIR = BASE / "runs_compare"                 # sama dgn cmp_yolo11m
NORM = BASE / "_pio_yolo"                         # struktur images/labels + dataset.yaml
YAML_PATH = NORM / "dataset.yaml"

# >>> Rilis RECTIFIED PIO — ditambahkan ke TRAIN saja (val tetap PIO asli) <<<
# SESUAIKAN path ini (tetap di E:\CCTV\PIO). Struktur diharapkan format YOLO:
#     <dir>/images/*.jpg  dan  <dir>/labels/*.txt   (1 kelas, konsisten dgn PIO)
# Set ke None untuk menonaktifkan.
RECTIFIED_PIO_DIR = BASE / "rectified_pio"        # <-- SESUAIKAN (mis. BASE / "PIO_rectified")

USE_OCCLUSION = True     # True = Run B (+oklusi). False = Run A (baseline).
EPOCHS = 100
IMGSZ = 960
LR0 = 0.02               # nilai paper (Tabel 7). Bila batch dinaikkan jauh & terlihat
                         # divergen, turunkan sedikit.
BATCH = 16               # RTX 5090 32GB. Boleh dinaikkan (mis. 32), TAPI Run A & B
                         # HARUS pakai batch sama. (Utk apple-to-apple vs cmp_yolo11m: set 2.)
MOMENTUM = 0.9
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")
# ======================================================================


# ======================================================================
# OKLUSI — patch Ultralytics Albumentations agar menambah CoarseDropout
# (lubang kecil < ukuran ayam => oklusi PARSIAL, minim risiko ghost-label).
# ======================================================================
def _ensure_albumentations():
    """Pastikan albumentations ada; install otomatis bila belum (agar langsung run)."""
    try:
        import albumentations  # noqa: F401
        return True
    except Exception:
        print("[setup] albumentations belum terpasang — menginstall ...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "albumentations>=1.4.0"],
                           check=True)
            import importlib
            importlib.invalidate_caches()
            import albumentations  # noqa: F401
            print("[setup] albumentations terpasang.")
            return True
        except Exception as e:
            print(f"[setup] gagal menyiapkan albumentations: {e}")
            return False


def _make_coarse_dropout(A, imgsz):
    """Bangun CoarseDropout secara robust lintas-versi albumentations (1.3 s/d 2.x)."""
    hmin, hmax = max(1, round(0.02 * imgsz)), max(2, round(0.05 * imgsz))
    attempts = [
        # albumentations >= 2.0 (fill)
        lambda: A.CoarseDropout(num_holes_range=(6, 16), hole_height_range=(0.02, 0.05),
                                hole_width_range=(0.02, 0.05), fill=0, p=0.5),
        # albumentations 1.4.x (num_holes_range + fill_value)
        lambda: A.CoarseDropout(num_holes_range=(6, 16), hole_height_range=(0.02, 0.05),
                                hole_width_range=(0.02, 0.05), fill_value=0, p=0.5),
        # albumentations < 1.4 (API lama, piksel)
        lambda: A.CoarseDropout(max_holes=16, min_holes=6, max_height=hmax, min_height=hmin,
                                max_width=hmax, min_width=hmin, fill_value=0, p=0.5),
    ]
    for i, f in enumerate(attempts):
        try:
            t = f()
            print(f"[occlusion] CoarseDropout dibangun (signature #{i+1}).")
            return t
        except Exception:
            continue
    return None


def enable_occlusion(imgsz=960):
    """Monkey-patch aman: bila apa pun gagal, training tetap jalan tanpa oklusi albumentations."""
    if not _ensure_albumentations():
        print("[occlusion] tanpa albumentations — hanya mixup/mosaic (native) yang aktif.")
        return False
    import ultralytics.data.augment as aug
    _orig_init = aug.Albumentations.__init__

    def _patched_init(self, p=1.0):
        _orig_init(self, p)                     # bangun transform default dulu
        try:
            import albumentations as A
            if self.transform is None:          # albumentations gagal load di init asli
                return
            occ = _make_coarse_dropout(A, imgsz)
            if occ is None:
                print("[occlusion] CoarseDropout tak terbentuk (versi albumentations?) — skip.")
                return
            T = list(self.transform.transforms) + [occ]
            self.contains_spatial = True        # CoarseDropout = spatial -> WAJIB bbox_params
            self.transform = A.Compose(
                T, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))
            if hasattr(self.transform, "set_random_seed"):
                self.transform.set_random_seed(torch.initial_seed())
            print("[occlusion] aktif ->", [t.__class__.__name__ for t in T])
        except Exception as e:
            print(f"[occlusion] patch gagal ({e}) — lanjut tanpa oklusi albumentations.")

    aug.Albumentations.__init__ = _patched_init
    print("[occlusion] patch terpasang (CoarseDropout lubang-kecil, p=0.5).")
    return True


# ======================================================================
# Dataset PIO: pakai ulang split lama bila ada; else bangun (logika notebook)
# ======================================================================
def _all_images(root):
    return sorted(p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True)
                  if p.lower().endswith(IMG_EXT))


def _find_split_dirs(root):
    f = {}
    for s in ("train", "val", "valid"):
        for d in glob.glob(os.path.join(root, "**", "images", s), recursive=True):
            f.setdefault(("images", "val" if s.startswith("val") else "train"), d)
        for d in glob.glob(os.path.join(root, "**", "labels", s), recursive=True):
            f.setdefault(("labels", "val" if s.startswith("val") else "train"), d)
    return f


def _label_for(img):
    stem = os.path.splitext(os.path.basename(img))[0]
    cand = os.path.splitext(img.replace(os.sep + "images" + os.sep,
                                        os.sep + "labels" + os.sep))[0] + ".txt"
    if os.path.exists(cand):
        return cand
    h = glob.glob(os.path.join(str(DATA_ROOT), "**", stem + ".txt"), recursive=True)
    return h[0] if h else None


def _parse_house(name):
    m = re.search(r"(?i)(?:^|[_\-])([CP])-?W[-_ ]?[1-6]", os.path.basename(name))
    return {"c": "Commercial", "p": "Prototype"}[m.group(1).lower()] if m else "Unknown"


def _parse_week(name):
    m = re.search(r"(?i)[-_ ]W[-_ ]?([1-6])\b", os.path.basename(name))
    return f"W{m.group(1)}" if m else "W?"


def _link(src, dst):
    if os.path.lexists(dst):
        return
    try:
        os.symlink(os.path.abspath(src), dst)
    except Exception:
        shutil.copy(src, dst)


def build_dataset():
    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset PIO tidak ada di {DATA_ROOT}. Ekstrak dulu seperti di notebook.")
    splits = _find_split_dirs(str(DATA_ROOT))
    premade = ("images", "train") in splits and ("images", "val") in splits
    recs = []
    if premade:
        for sp in ("train", "val"):
            for img in _all_images(splits[("images", sp)]):
                recs.append((img, _label_for(img), sp))
    else:
        pairs = [(i, _label_for(i)) for i in _all_images(str(DATA_ROOT))]
        pairs = [(i, l) for i, l in pairs if l]
        rng = random.Random(SEED)
        st = collections.defaultdict(list)
        for i, l in pairs:
            st[(_parse_house(i), _parse_week(i))].append((i, l))
        for items in st.values():
            items = sorted(items)
            rng.shuffle(items)
            nval = round(len(items) * 0.3)
            for idx, (i, l) in enumerate(items):
                recs.append((i, l, "val" if idx < nval else "train"))

    ntr = sum(1 for r in recs if r[2] == "train")
    nva = sum(1 for r in recs if r[2] == "val")
    print(f"[data] split PIO: {'BAWAAN' if premade else '70/30 dibuat'} | train {ntr} | val {nva}")

    shutil.rmtree(str(NORM), ignore_errors=True)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(str(NORM), sub), exist_ok=True)
    for img, lbl, sp in recs:
        if not lbl:
            continue
        base = os.path.basename(img)
        stem = os.path.splitext(base)[0]
        _link(img, os.path.join(str(NORM), "images", sp, base))
        _link(lbl, os.path.join(str(NORM), "labels", sp, stem + ".txt"))
    yaml.safe_dump({"path": str(NORM.resolve()), "train": "images/train", "val": "images/val",
                    "nc": 1, "names": ["pollo"]}, open(str(YAML_PATH), "w"), sort_keys=False)
    print(f"[data] dataset.yaml siap: {YAML_PATH}")


def build_training_yaml():
    """Tulis yaml training: val = PIO asli; train = PIO asli (+rectified bila ada). Return (path, used_rect)."""
    train_dirs = [str((NORM / "images" / "train").resolve())]
    used_rect = False
    if RECTIFIED_PIO_DIR is not None:
        rp = Path(RECTIFIED_PIO_DIR)
        img_dir = rp / "images" if (rp / "images").exists() else rp
        if img_dir.exists():
            n = len([p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXT])
            if n == 0:
                print(f"[data] !! RECTIFIED_PIO_DIR ada tapi tak ada gambar: {img_dir} → dilewati")
            else:
                train_dirs.append(str(img_dir.resolve()))
                used_rect = True
                print(f"[data] +rectified PIO: {img_dir} ({n} gambar) → ditambah ke TRAIN")
        else:
            print(f"[data] !! RECTIFIED_PIO_DIR di-set tapi tak ada: {rp} → dilewati "
                  f"(sesuaikan path-nya, struktur <dir>/images + <dir>/labels)")
    out = NORM / "dataset_train.yaml"
    yaml.safe_dump({"path": str(NORM.resolve()), "train": train_dirs,
                    "val": str((NORM / "images" / "val").resolve()),
                    "nc": 1, "names": ["pollo"]}, open(str(out), "w"), sort_keys=False)
    print(f"[data] training yaml: {out} (train paths: {len(train_dirs)}, rectified={used_rect})")
    return out, used_rect


# ======================================================================
def main():
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) dataset PIO
    if YAML_PATH.exists() and (NORM / "images" / "train").exists() and (NORM / "images" / "val").exists():
        print(f"[data] pakai ulang split PIO yang ada (sama dgn cmp_yolo11m): {YAML_PATH}")
    else:
        print("[data] membangun split PIO ...")
        build_dataset()
    train_yaml, used_rect = build_training_yaml()

    # 2) oklusi
    if USE_OCCLUSION:
        enable_occlusion(IMGSZ)

    # 3) nama run otomatis
    parts = ["cmp_yolov8m"]
    if USE_OCCLUSION:
        parts.append("occ")
    if used_rect:
        parts.append("rect")
    run_name = "_".join(parts)

    print(f"ultralytics {ultralytics.__version__} | torch {torch.__version__} | "
          f"CUDA={torch.cuda.is_available()} | "
          f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if not torch.cuda.is_available():
        print("!! PERINGATAN: CUDA tak terdeteksi — 960px di CPU akan sangat lambat.")
    print(f"[run] {run_name} | epochs={EPOCHS} imgsz={IMGSZ} batch={BATCH} lr0={LR0} "
          f"occlusion={USE_OCCLUSION}")

    # 4) training — resep Tabel 7 tier-m
    model = YOLO("yolov8m.pt")
    model.train(
        data=str(train_yaml), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
        optimizer="AdamW", lr0=LR0, momentum=MOMENTUM,
        seed=SEED, deterministic=True,
        mosaic=1.0, close_mosaic=10,
        mixup=(0.10 if USE_OCCLUSION else 0.0),   # blend = oklusi transparan (Run B)
        project=str(RUNS_DIR), name=run_name, exist_ok=True, verbose=True,
    )

    # 5) validasi in-domain (val = PIO asli)
    best = RUNS_DIR / run_name / "weights" / "best.pt"
    m = YOLO(str(best)).val(data=str(train_yaml), split="val", imgsz=IMGSZ, batch=BATCH, verbose=False)
    print(f"\n=== {run_name} — in-domain PIO-val ===")
    print(f"P={float(m.box.mp):.3f}  R={float(m.box.mr):.3f}  "
          f"mAP50={float(m.box.map50):.3f}  mAP50-95={float(m.box.map):.3f}")
    print("Ref cmp_yolo11m : P=0.958 R=0.888 mAP50=0.935 mAP50-95=0.766")
    print("Ref paper v10m  : P=0.961 R=0.880 mAP50=0.970 mAP50-95=0.760")
    print(f"\nBobot terbaik: {best}\nALL DONE")


if __name__ == "__main__":
    main()
