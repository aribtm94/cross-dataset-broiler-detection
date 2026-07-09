# Cross-Dataset Generalization of Broiler Weight Estimation & Anomaly Detection

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10-blue.svg">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.1.2%2Bcu121-ee4c2c.svg">
  <img alt="YOLO" src="https://img.shields.io/badge/detector-Ultralytics%20YOLO-00b8d4.svg">
  <img alt="License" src="https://img.shields.io/badge/code-MIT-green.svg">
  <img alt="MOWA" src="https://img.shields.io/badge/MOWA-S--Lab%20NC-orange.svg">
</p>

An undergraduate thesis (*skripsi*) research pipeline that measures how well a
broiler-chicken detector **generalizes across cameras and datasets**, and detects
**weight anomalies** in flocks — without per-dataset ground-truth weights.

The headline experiment asks a concrete question: *does end-to-end fisheye/
wide-angle rectification (MOWA) as a preprocessing step improve cross-dataset
detection?* The answer, rigorously measured, is a **valid negative result** —
documented in full below.

> **License note:** the MOWA model this project evaluates is under the **S-Lab
> License 1.0 (non-commercial)**. The pipeline as bundled is for research /
> education use only. See [`NOTICE`](NOTICE) and
> [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md).

---

## Pipeline overview

1. **Detect** broilers with a YOLO model trained on the PIO dataset.
2. **Rectify** each image with **MOWA** (end-to-end warping, TPAMI 2025) as
   preprocessing, then re-evaluate to see whether mAP improves (**A/B test**).
3. **Estimate relative weight** of each bounding box against the Cobb500 growth
   standard.
4. **Detect weight anomalies** with an **unsupervised voting ensemble** (adapted
   from a cattle-outlier method), compared against a percentile baseline.
5. **Explore** every dataset's results in a **Streamlit dashboard**.

```
                 ┌─────────────┐   A/B preprocessing    ┌──────────────┐
   images  ────► │  YOLO detect│ ◄────────────────────► │ MOWA rectify │
                 └──────┬──────┘   (+ label warp)        └──────────────┘
                        │
                        ▼
              bbox features ──► relative weight (Cobb500) ──► anomaly ensemble
                        │                                            │
                        └───────────────► reports/ ◄────────────────┘
                                             │
                                             ▼
                                   Streamlit dashboard
```

---

## Repository layout

```
src/                     # Pipeline scripts (detection, eval, weight, anomaly)
  eval_detection.py        # YOLO mAP on PIO + external datasets
  mowa_rectify.py          # MOWA preprocessing (rectify image + warp labels)
  finetune_rectified.py    # "rectify-both" fine-tune on rectified domain
  compare_ab.py            # baseline vs MOWA A/B comparison + verdict
  extract_bbox_features.py # per-bbox geometric features
  estimate_weight_anomalies.py
  anomaly_ensemble.py      # unsupervised voting ensemble (new method)
  anomaly_compare.py       # ensemble vs percentile → recommendation
  common.py                # shared utils (paths, Cobb500, stats, IO)
  ...                      # ~40 scripts; see docs/ for the full campaign
dashboard/app.py         # Streamlit dashboard
configs/                 # dataset configs, Cobb500 table, calibration
scripts/                 # env setup + asset download/build tooling
docs/                    # research notes, methodology reviews, checkpoints
reports/                 # committed small result summaries (CSV/JSON/HTML)
assets_manifest.json     # single source of truth for the Google-Drive bundles

# fetched separately (see docs/DATA_SETUP.md) — not in Git:
data/                    # datasets + rectified/enhanced/augmented outputs
train model/             # trained YOLO weights + runs
vendor/MOWA/             # vendored MOWA model + checkpoint
features/                # large intermediate CSVs
```

Large folders (`data/`, `train model/`, `vendor/`, `features/`, `.venv*`) are
intentionally **git-ignored**. The repo holds the code and small result
summaries; everything heavy is downloaded via the asset bundles.

---

## Quickstart

```powershell
# 1. clone
git clone https://github.com/aribtm94/cross-dataset-broiler-detection.git
cd cross-dataset-broiler-detection

# 2. create both Python 3.10 venvs with CUDA 12.1 torch (needs an NVIDIA GPU)
./scripts/setup_env.ps1                       # bash: ./scripts/setup_env.sh

# 3. clone the MOWA model into vendor/ (or get it via the mowa-sam bundle)
git clone https://github.com/KangLiao929/MOWA vendor/MOWA

# 4. download datasets + weights + MOWA checkpoint from Google Drive
./.venv-mowa/Scripts/python.exe scripts/download_assets.py --required

# 5. run the baseline evaluation
./.venv-yolo/Scripts/python.exe src/eval_detection.py `
    --weights "train model/runs_compare/cmp_yolov8m/weights/best.pt" `
    --out reports/eval_baseline.json
```

See [`docs/DATA_SETUP.md`](docs/DATA_SETUP.md) for all download links and layout,
and the full pipeline below.

---

## 1. Environments (two separate venvs)

Two Python 3.10 virtual environments keep the dependencies from clashing:

| venv | used for | key packages |
|------|----------|--------------|
| `.venv-yolo` | YOLO eval, weight/anomaly pipeline, dashboard | ultralytics, torch cu121, streamlit |
| `.venv-mowa` | MOWA rectification inference, asset tooling | torch cu121, timm, einops, gdown |

**A CUDA 12.1 GPU is required** (MOWA hard-codes `.cuda()`). Tested on an
RTX 4060 Laptop 8 GB, CUDA 12.1, Windows 11.

Automated setup:

```powershell
./scripts/setup_env.ps1           # Windows
./scripts/setup_env.sh            # Linux / macOS / Git-Bash
```

Manual setup (if you prefer): install CUDA torch **first**, then the rest.

```bash
python -m venv .venv-yolo
.venv-yolo/Scripts/python.exe -m pip install torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu121
.venv-yolo/Scripts/python.exe -m pip install -r requirements-yolo.txt

python -m venv .venv-mowa
.venv-mowa/Scripts/python.exe -m pip install torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu121
.venv-mowa/Scripts/python.exe -m pip install -r requirements-mowa.txt
```

> **Order matters:** finish installing CUDA torch before `timm`/`ultralytics`.
> Installing them first can pull a CPU torch build and overwrite the CUDA one.

---

## 2. Data & weights

Everything heavy is on Google Drive and described in
[`docs/DATA_SETUP.md`](docs/DATA_SETUP.md). The fast path:

```powershell
# datasets + trained weights + MOWA/SAM (the required set, ~4.7 GB)
./.venv-mowa/Scripts/python.exe scripts/download_assets.py --required

# or absolutely everything, incl. derived data + features + papers (~16 GB)
./.venv-mowa/Scripts/python.exe scripts/download_assets.py --all
```

The downloader reads [`assets_manifest.json`](assets_manifest.json), verifies each
zip's SHA-256, and extracts it to the correct folder. External datasets can
alternatively be re-pulled from Roboflow:

```bash
.venv-yolo/Scripts/python.exe src/download_roboflow_datasets.py --api-key <ROBOFLOW_KEY>
```

(Use a free Roboflow key; never commit it — copy `.env.example` to `.env`.)

Label format is standard YOLO: `class cx cy w h` (normalized).

---

## 3. Running the pipeline

All commands from the repo root.

### Step 1 — Baseline evaluation (condition A)
```bash
.venv-yolo/Scripts/python.exe src/eval_detection.py \
    --weights "train model/runs_compare/cmp_yolov8m/weights/best.pt" \
    --out reports/eval_baseline.json
```

### Step 2 — MOWA rectification + label warp (all datasets)
```bash
# PIO val
.venv-mowa/Scripts/python.exe src/mowa_rectify.py \
    --input data/images/val --labels data/labels/val \
    --output data/rectified/pio_val --label-mode warp
# broiler_instance_seg
.venv-mowa/Scripts/python.exe src/mowa_rectify.py \
    --input data/external/broiler_instance_seg/train/images \
    --labels data/external/broiler_instance_seg/train/labels \
    --output data/rectified/broiler_instance_seg --label-mode warp
# chicken_detection_fum (repeat for test/valid/train into the same output)
.venv-mowa/Scripts/python.exe src/mowa_rectify.py \
    --input data/external/chicken_detection_fum/test/images \
    --labels data/external/chicken_detection_fum/test/labels \
    --output data/rectified/chicken_detection_fum --label-mode warp
```
`--label-mode warp` transforms each box to follow the rectification
(instance-mask + nearest), so labels stay aligned with the MOWA output.
~2–5 s/image on an RTX 4060.

### Step 3 — MOWA evaluation (condition B) + A/B comparison
```bash
.venv-yolo/Scripts/python.exe src/eval_detection.py \
    --weights "train model/runs_compare/cmp_yolov8m/weights/best.pt" \
    --rectified-root data/rectified --out reports/eval_mowa.json
.venv-yolo/Scripts/python.exe src/compare_ab.py     # -> reports/ab_comparison.{json,csv,html}
```

### Step 3b — (If MOWA is worse) "rectify-both" fine-tune → condition B′
If `compare_ab` returns the verdict **mowa_worse**, that is expected: the
detector was trained on original images but tested on rectified ones (domain
mismatch). The literature fix (KITTI-360 fisheye benchmark; FisheyeYOLO/
WoodScape) is to fine-tune the detector on the rectified domain.
```bash
# also rectify the PIO train set
.venv-mowa/Scripts/python.exe src/mowa_rectify.py \
    --input data/images/train --labels data/labels/train \
    --output data/rectified/pio_train --label-mode warp
# fine-tune YOLOv8m on rectified train+val
.venv-yolo/Scripts/python.exe src/finetune_rectified.py --epochs 40
# re-evaluate the fine-tuned weights (condition B′) + compare
.venv-yolo/Scripts/python.exe src/eval_detection.py \
    --weights "train model/runs_rectified/ft_rectified_yolov8m/weights/best.pt" \
    --rectified-root data/rectified --out reports/eval_mowa_ft.json
.venv-yolo/Scripts/python.exe src/compare_ab.py --mowa reports/eval_mowa_ft.json \
    --out-prefix reports/ab_comparison_ft
```

### Step 4 — Weight features + anomaly detection
```bash
.venv-yolo/Scripts/python.exe src/extract_bbox_features.py
.venv-yolo/Scripts/python.exe src/estimate_weight_anomalies.py
.venv-yolo/Scripts/python.exe src/compare_camera_corrections.py
.venv-yolo/Scripts/python.exe src/anomaly_ensemble.py     # -> reports/anomaly_ensemble_*
.venv-yolo/Scripts/python.exe src/anomaly_compare.py      # -> reports/anomaly_method_comparison.*
```

### Step 5 — Dashboard
```bash
.venv-yolo/Scripts/python.exe -m streamlit run dashboard/app.py
```
Open the printed URL (default http://localhost:8501). Pick a dataset in the
sidebar and browse the **Baseline / MOWA / Anomaly / Metrics** tabs.

---

## 4. Key outputs

| File | Contents |
|------|----------|
| `reports/eval_baseline.json` / `eval_mowa.json` | per-dataset mAP (A / B) |
| `reports/ab_comparison.html` | A/B table + verdict |
| `reports/experiments_v2_master.html` | master table of every MOWA variant tried |
| `reports/anomaly_ensemble_report.html` | voting-ensemble summary |
| `reports/anomaly_method_comparison.html` | ensemble vs percentile + recommendation |
| `reports/anomaly_review_sample.csv` | top-scoring boxes for manual review |

---

## 5. Results — MOWA A/B (summary)

Model: YOLOv8m trained on PIO. Primary metric: mAP@50-95. Three conditions:

| Dataset | A: baseline | B: MOWA as-is | B′: MOWA + fine-tune |
|---------|-------------|---------------|-----------------------|
| PIO val | 0.710 | 0.638 | **0.683** |
| broiler_instance_seg | 0.536 | 0.457 | **0.530** |
| chicken_detection_fum | 0.058 | 0.049 | **0.058** |
| **mean Δ vs A** | — | **−0.053** | **−0.011** |

**Conclusion:** MOWA rectification as preprocessing **does not beat the
baseline**. Tested as-is, MOWA is worse (−0.053) because the detector was trained
on original images (domain mismatch). After "rectify-both" fine-tuning, ~79 % of
the loss is recovered (−0.011) but it still sits just under baseline. This is
consistent with the fisheye literature (WoodScape / FisheyeYOLO): for mildly
distorted cameras, undistortion does not reliably help modern detectors. **A
valid negative result** for the thesis. Additional soften / coarse / pad / SAM
variants were later screened (see `reports/experiments_v2_master.html`); the best
(`pad015` fine-tune) ties the baseline rather than beating it.

---

## 6. Methodology notes

- **Fair A/B:** labels are warped (mode `warp`), so the mAP change reflects the
  rectification effect, not label misalignment. Boxes warped out of frame
  (FOV trim) are dropped.
- **External datasets are a different domain** from the PIO training data; low
  absolute numbers are a generalization signal, not a bug.
- **Anomaly without ground truth:** "best" is judged by rate stability and
  inter-method agreement, not accuracy against anomaly labels (which do not
  exist for these flocks).

Deeper write-ups live in [`docs/`](docs/) — methodology reviews, the MOWA v2
experiment plan, and cross-dataset generalization tables.

---

## 7. Citation

If this work is useful, please cite it (and the components it builds on):

```bibtex
@misc{generalisasi_ayam_skripsi,
  title  = {Cross-Dataset Generalization of Broiler Weight Estimation and Anomaly Detection},
  author = {Arib},
  year   = {2026},
  note   = {Undergraduate thesis pipeline. Uses MOWA (S-Lab NC) and Ultralytics YOLO.},
  howpublished = {\url{https://github.com/aribtm94/cross-dataset-broiler-detection}}
}
```

See [`CITATION.cff`](CITATION.cff) for machine-readable metadata.

---

## 8. License & credits

- **Own code:** [MIT](LICENSE).
- **MOWA:** [KangLiao929/MOWA](https://github.com/KangLiao929/MOWA), **S-Lab
  License 1.0 (non-commercial)**.
- **YOLO:** [Ultralytics](https://github.com/ultralytics/ultralytics), AGPL-3.0.
- **MobileSAM:** Apache-2.0.
- **Weight standard:** Cobb500 Broiler Performance Supplement 2022 (© Cobb-Vantress).
- **External datasets:** Roboflow Universe (see `data/external/*/README.roboflow.txt`).

Full attribution and restrictions: [`NOTICE`](NOTICE) and
[`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md).
