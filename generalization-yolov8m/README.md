# Cross-Dataset Generalization of a Broiler Detector (YOLOv8m)

Sub-project of **cross-dataset-broiler-detection**. This folder studies **how well a
broiler-chicken detector trained on one dataset (PIO) generalizes to *unseen*
cameras and datasets**, with no target-domain fine-tuning. It is a self-contained
evaluation + analysis study; it shares the repository's datasets, configs and
license files with the other sub-projects.

> **Status:** research code + result summaries. Trained weights and raw images are
> *not* in Git (see [Data & weights](#data--weights)). Numbers below are read from
> [`reports/generalization_eval_summary.json`](reports/generalization_eval_summary.json).

---

## Research question

A YOLO detector trained only on the in-house **PIO** broiler dataset is evaluated,
**without any adaptation**, on five external chicken/broiler datasets that differ in
camera, lighting, housing, breed age and — critically — **annotation protocol**. The
question: *where does an in-domain detector still work off-domain, and what actually
drives the failures — the model, or the label/domain shift?*

## Place in the monorepo

```
<repo-root>/
├─ data/               ← SHARED datasets (git-ignored, fetched via manifest)
├─ configs/            ← SHARED dataset configs (dataset.yaml, cobb500, external registry)
├─ assets_manifest.json, scripts/download_assets.py   ← SHARED asset tooling
├─ LICENSE, NOTICE, THIRD_PARTY_LICENSES.md, CITATION.cff
│
└─ generalization-yolov8m/     ← THIS sub-project
   ├─ src/          8 evaluation / rendering drivers
   ├─ docs/         study plan + literature notes
   └─ reports/      small result summaries (the numbers that go in the paper)
```

This folder reuses the shared `data/` and `configs/`. It does **not** duplicate the
dataset or the common pipeline scripts (`extract_bbox_features.py`,
`estimate_weight_anomalies.py`, `common.py`, …) that already live in the repo's root
`src/`.

---

## Datasets

| Key | Role | Source | License |
|-----|------|--------|---------|
| `pio_original_val` | in-domain reference | PIO broiler dataset — Zenodo `10.5281/zenodo.16686320`; paper DOI `10.1038/s41597-026-07114-5` | CC-BY 4.0 |
| `broiler_instance_seg` | external | Roboflow Universe `broiler-data/broiler-ozg7f` | per-uploader (see page) |
| `chicken_detection_fum` | external | Roboflow Universe `fum-icce/chicken-detection-z6wni` | per-uploader (see page) |
| `broiler_healthy_sick` | external | Roboflow Universe `technicalresearch/broiler-chicken-healthy-and-sick` | per-uploader (see page) |
| `chicken_count` | external | Roboflow Universe `chickendetection-sct5j/chicken-count` | per-uploader (see page) |
| `nestler_yolo` | external | NESTLER — Zenodo `10.5281/zenodo.20924893` (converted video→YOLO bbox) | CC-BY 4.0 |

Images are **not redistributed** here. Fetch them via the shared tooling / the
Roboflow downloader (`src/download_roboflow_datasets.py` in the repo root, using your
own `ROBOFLOW_API_KEY`) and the Zenodo DOIs. Only *our derivative labels* and the
external-dataset registry (`configs/datasets/external_datasets.json`) are shared.

---

## Evaluation protocol

- **Detector:** YOLOv8m trained on PIO (thesis focus). A YOLOv8..v12 comparison also
  exists; see the checkpoint note below.
- **Inference:** `imgsz=960` (matches training), `batch=1`, per-image streaming to fit
  a low-RAM host. **On a per-image CUDA-OOM the script falls back to `imgsz=640` for
  that image only** — disclose this, as a run on a memory-starved machine can mix 960
  and 640 inference. `conf`/IoU use Ultralytics defaults (`val` conf ≈ 0.001, `predict`
  conf 0.25) unless you set them.
- **Metrics:** Ultralytics box metrics — precision (`box.mp`), recall (`box.mr`),
  mAP@50 (`box.map50`), mAP@50-95 (`box.map`, COCO-style).
- **Val splits:** PIO uses its real `val` split (452 imgs). `broiler_instance_seg` ships
  no val folder, so a **deterministic 20% train-as-val split (`seed=42`)** is used —
  regenerated on the fly, identical across runs. `chicken_detection_fum` uses its real
  `valid/` split (18 imgs).

## How to run

> **Path assumptions.** These drivers were the exact scripts run on the eval machine.
> Each resolves paths relative to its own location and expects this working layout:
> weights at `runs_compare/<model>/weights/best.pt`, the PIO dataset at `_pio_yolo/`,
> and the external datasets at `../data/data/external/`. To reproduce, either recreate
> that layout from the downloaded bundle **or** edit the `MODEL_PATH` / `EXTERNAL_DIR` /
> `PIO` constants at the top of each script. *(Follow-up: lift these to env vars / CLI
> args for a cleaner public API.)*

```bash
# in-domain baseline (PIO val)              -> reports/generalization_eval_summary.json
python src/run_pio_original.py
# cross-dataset external eval (full sets)   -> summary.json
python src/run_external_eval.py
# val-split-only re-test with YOLOv8m       -> summary.json["val_only"]
python src/run_val_only_eval.py
# occlusion-augmented YOLOv8m variant       -> summary.json["val_only_occ"]
python src/run_val_only_eval_occ.py
# isolate model vs split for the FUM result
python src/verify_fum_valonly.py
# qualitative figures (GT / pred / overlay)
python src/render_bbox_gt.py && python src/render_bbox_occ.py && python src/render_bbox_overlay.py
```

Dependencies: install the sub-project's own pinned set — the **exact eval-host
environment** — into a fresh Python 3.13 venv. Install `torch` first, then the rest:

```bash
# STEP 1 — exact reproduction: the numbers were computed on CPU (torch 2.12.1+cpu)
pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cpu
#           (or the cu121 index for optional GPU inference — see requirements.txt header)
# STEP 2
pip install -r generalization-yolov8m/requirements.txt   # ultralytics 8.4.84, etc.
```

This set is intentionally self-contained — **not** the repo-root `requirements-yolo.txt`
(older stack for the other sub-projects; would conflict, e.g. numpy 2.5 vs 1.26).
**Evaluation ran on CPU** (`batch=1`, low-RAM host), so the `imgsz` OOM fallback in the
protocol above did not trigger. Model *training* was done separately on a CUDA GPU with
a slightly different stack (ultralytics 8.3.152 / torch 2.11.0+cu128 / Python 3.11).

---

## Results

Cross-dataset detection, model trained on PIO only (from
[`reports/generalization_eval_summary.json`](reports/generalization_eval_summary.json),
full-set run):

| Dataset | Images | P | R | mAP@50 | mAP@50-95 |
|---------|-------:|----:|----:|-------:|----------:|
| PIO val (in-domain) | 452 | 0.958 | 0.887 | 0.899 | **0.709** |
| broiler_instance_seg | 200 | 0.864 | 0.767 | 0.836 | **0.579** |
| chicken_detection_fum | 326 | 0.554 | 0.151 | 0.139 | **0.061** |
| nestler_yolo | 480 | 0.000 | 0.000 | 0.000 | **0.000** |

**Reading the numbers**

- **broiler_instance_seg — generalizes well.** ~0.58 mAP@50-95 off-domain with zero
  adaptation is a genuinely strong transfer result.
- **chicken_detection_fum — misleadingly low.** The model *detects* birds but boxes are
  loose against this dataset's tighter annotation convention, so mAP collapses while
  recall/precision on the easier `valid/` subset recover sharply. `verify_fum_valonly.py`
  runs YOLOv8m **and** YOLO11m on the *same* 18 valid images to show the jump is driven
  by the **split / annotation protocol, not the model swap**. This is an
  annotation-shift finding, not a pure detector failure (cf. the label-protocol
  literature in [`docs/RENCANA_GENERALISASI_YOLOV8M.md`](docs/RENCANA_GENERALISASI_YOLOV8M.md)).
- **nestler_yolo — total failure (0.0).** A real, verified out-of-distribution wall
  (top-view/behaviour footage), not a bug.

Cross-dataset relative-weight/anomaly comparisons and the domain-shift analysis are in
[`reports/cross_dataset_relative_summary.*`](reports/) and
[`reports/dense_domain_shift_analysis.*`](reports/).

> ⚠️ **Checkpoint provenance (must fix before submission).** The full-set table above
> was produced with the **YOLO11m** checkpoint (`cmp_yolo11m/weights/best.pt`), while
> the val-only re-test (`summary.json["val_only"]`) uses **YOLOv8m**
> (`cmp_yolov8m/weights/best.pt`) — the thesis-focus model. **TODO(you):** pick the one
> authoritative checkpoint for the paper, report a single consistent set of numbers,
> and publish that checkpoint's SHA-256 + download link.

---

## Data & weights

Nothing heavy is committed. Reproduce via three tiers:

1. **In Git (here):** code, docs, and the small result summaries in `reports/`.
2. **Off-Git bundle:** trained weights + full result tables, listed in the repo-root
   `assets_manifest.json` with checksums, fetched by `scripts/download_assets.py`.
   ⚠️ Ultralytics YOLO is **AGPL-3.0**, so every trained `*.pt` is an AGPL-derivative —
   publish weights as a **separate, explicitly AGPL-3.0 artifact**, not inside this
   (MIT) code tree. **TODO(you):** add the generalization weights to the manifest and
   mint a Zenodo DOI for them.
3. **Original images:** fetched by the user from the Zenodo DOIs / Roboflow (with your
   own API key). Not re-hosted.

## Files in this folder

- `src/` — `run_pio_original.py` (in-domain baseline), `run_external_eval.py` (full-set
  cross-dataset eval), `run_val_only_eval.py` (YOLOv8m val-only), `run_val_only_eval_occ.py`
  (occlusion-aug variant), `verify_fum_valonly.py` (model-vs-split control),
  `render_bbox_{gt,occ,overlay}.py` (qualitative figures).
- `docs/` — `RENCANA_GENERALISASI_YOLOV8M.md` (study plan + DG literature),
  `RESEARCH_END_TO_END_PIPELINE.md` (pipeline design notes),
  `RANGKUMAN_PAPER_REFERENSI.txt` (reference summaries).
- `reports/` — `generalization_eval_summary.json` (the mAP numbers),
  `cross_dataset_relative_summary.*`, `dense_domain_shift_analysis.*`,
  `external_dataset_summary.csv`, and per-dataset `external/*_audit.json`,
  `external/*_image_stats.csv`, `external/<ds>/relative_anomaly_summary.json`.

## License & citation

Own code: **MIT** (repo-root `LICENSE`). Third-party components (Ultralytics YOLO
AGPL-3.0; PIO / NESTLER CC-BY; Roboflow per-dataset; Cobb500 growth standard):
repo-root [`NOTICE`](../NOTICE) and [`THIRD_PARTY_LICENSES.md`](../THIRD_PARTY_LICENSES.md).
Cite via repo-root `CITATION.cff` plus the PIO paper (`10.1038/s41597-026-07114-5`) and
the dataset DOIs above.

---