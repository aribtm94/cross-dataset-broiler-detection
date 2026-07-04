# Data, Weights & Assets Setup

The heavy parts of this project — datasets, trained weights, the MOWA model, and
derived outputs — are **too large for Git** and live on Google Drive. This page
is the one place that maps every bundle to its download link and target folder.

There are two ways to get the assets: the **automated** downloader (recommended)
or **manual** download + unzip.

---

## 0. Prerequisites

- The two virtual environments created (see the main [README](../README.md) §2
  or run `scripts/setup_env.ps1`). The downloader uses `gdown`, which is in
  `.venv-mowa`.
- ~5 GB free disk for the required bundles (they extract to ~5 GB more), or
  ~16 GB download / ~35 GB extracted if you also fetch the optional derived
  data / features / papers.

---

## 1. Download links

> **Maintainer:** after running `scripts/build_release_bundles.py` and uploading
> each `dist/*.zip` to Google Drive (share = *anyone with the link*), paste the
> links into **both** the table below **and** the matching `gdrive_url` field in
> [`assets_manifest.json`](../assets_manifest.json). The automated downloader
> reads the manifest; this table is for humans.

| Bundle | Required | Contents | Extracts to | Link |
|--------|----------|----------|-------------|------|
| `datasets-core.zip` | ✅ | PIO images/labels + external Roboflow datasets, `classes.txt`, `dataset.yaml`, `FilePrefixCode.xlsx` | `data/` | _paste link_ |
| `weights-yolo.zip` | ✅ | All YOLO training runs + best/last checkpoints + base weights | `train model/` | _paste link_ |
| `mowa-sam.zip` | ✅ | Vendored MOWA repo + `mowa_pretrained.pth` + `mobile_sam.pt` | `vendor/MOWA/`, `mobile_sam.pt` | _paste link_ |
| `derived-data.zip` | ⬜ | Rectified / enhanced / augmented / mask variants | `data/` | _paste link_ |
| `features.zip` | ⬜ | `bbox_features`, `weight_estimates*`, ensemble CSVs | `features/` | _paste link_ |
| `papers.zip` | ⬜ | Reference papers (PDF) | repo root + `papers/` | _paste link_ |
| `rilis_rectified_pio.zip` | ⬜ | Curated best rectified-PIO share package | `rilis_rectified_pio/` | _paste link_ |

Required bundles are everything you need to reproduce the evaluation. Optional
bundles just let you skip recomputation — every derived artifact can be
regenerated with the `src/` scripts documented in the README.

---

## 2. Automated download (recommended)

From the repo root, using the MOWA venv's Python (it has `gdown`):

```powershell
# Windows PowerShell
.\.venv-mowa\Scripts\python.exe scripts\download_assets.py --list        # see status
.\.venv-mowa\Scripts\python.exe scripts\download_assets.py --required    # datasets + weights + MOWA
.\.venv-mowa\Scripts\python.exe scripts\download_assets.py --all         # everything
```

```bash
# Linux / macOS / Git-Bash
.venv-mowa/bin/python scripts/download_assets.py --required
```

The script downloads each zip into `dist/`, verifies its SHA-256 against the
manifest, and extracts it to the right place. You can also grab specific
bundles:

```powershell
.\.venv-mowa\Scripts\python.exe scripts\download_assets.py --only datasets_core weights_yolo
```

Re-running is safe: a cached zip with a matching hash is reused instead of
re-downloaded.

---

## 3. Manual download

If you prefer to click through Google Drive:

1. Open each link in the table above and download the `.zip`.
2. Unzip it **at the repo root** so the internal paths land correctly. Every
   bundle stores repo-relative paths (e.g. `data/images/...`,
   `train model/...`), so extracting at the root is all that's needed.

   ```powershell
   Expand-Archive datasets-core.zip -DestinationPath . -Force
   Expand-Archive "weights-yolo.zip" -DestinationPath . -Force
   Expand-Archive mowa-sam.zip     -DestinationPath . -Force
   ```

---

## 4. Expected layout after setup

Once the required bundles are extracted you should have:

```
data/
  images/{train,val}/*.jpg
  labels/{train,val}/*.txt
  external/broiler_instance_seg/train/{images,labels}
  external/chicken_detection_fum/{test,valid,train}/{images,labels}
  classes.txt  dataset.yaml  FilePrefixCode.xlsx
train model/
  runs_compare/cmp_yolov8m/weights/best.pt        # primary detector
  runs_rectified/ft_rectified_yolov8m/weights/best.pt
  runs_pad015/ft_pad015_yolov8m/weights/best.pt
  ...
vendor/MOWA/
  checkpoint/mowa_pretrained.pth
  model/  test.py  LICENSE  ...
mobile_sam.pt
```

---

## 5. Alternative: rebuild from original sources (no Google Drive)

You can avoid the Drive bundles entirely for some assets:

- **MOWA**: `git clone https://github.com/KangLiao929/MOWA vendor/MOWA`, then
  download the checkpoint (Google Drive id `1fxQbD1TLoRnW8lG2a8KMinmD6Jlol8EX`)
  to `vendor/MOWA/checkpoint/mowa_pretrained.pth`.
- **External datasets**: `.venv-yolo/Scripts/python.exe src/download_roboflow_datasets.py --api-key <YOUR_ROBOFLOW_KEY>`
  (free Roboflow key; never commit it — use `.env`, see `.env.example`).
- **Derived data / features**: regenerate with the pipeline (README §4).

The PIO source images and the trained YOLO weights, however, are only available
through the Google Drive bundles above.

---

## 6. For maintainers — (re)building the bundles

```powershell
# build all dist/*.zip and refresh size + sha256 in the manifest
.\.venv-mowa\Scripts\python.exe scripts\build_release_bundles.py

# rebuild just one
.\.venv-mowa\Scripts\python.exe scripts\build_release_bundles.py --only datasets_core

# only re-hash existing zips (e.g. after a manual repack)
.\.venv-mowa\Scripts\python.exe scripts\build_release_bundles.py --hash-only
```

Then upload `dist/*.zip` to Google Drive and paste the links (this file's table
+ `assets_manifest.json`).
