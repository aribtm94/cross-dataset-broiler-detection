# NEXT STEPS — Fase 1.3 Download & Validasi Dataset

Status terakhir: shell Claude sempat `test is temporarily unavailable`, jadi command Python belum bisa dijalankan dari agent. File dan script sudah siap.

## 1. Inspect schema NESTLER

Jalankan dari root project:

```powershell
python scripts/prepare_nestler_dataset.py --inspect-only
```

Target input:

```text
data/external/nestler_poultry_behaviour.zip
```

Yang diharapkan:
- Menampilkan shape JSON annotations NESTLER
- Tidak mengekstrak video
- Jika schema terbaca, lanjut step 2

## 2. Convert NESTLER sample ke YOLO bbox

Pastikan `ffmpeg` tersedia:

```powershell
ffmpeg -version
ffprobe -version
```

Jika belum ada, install ffmpeg dulu. Jika sudah:

```powershell
python scripts/prepare_nestler_dataset.py --max-frames-per-video 80
```

Output:

```text
data/external/nestler_yolo/images/val/*.jpg
data/external/nestler_yolo/labels/val/*.txt
data/external/nestler_yolo/dataset.yaml
data/external/nestler_yolo/metadata.json
```

## 3. Validasi hasil konversi NESTLER

```powershell
python scripts/validate_external_dataset.py --name nestler_yolo --root data/external/nestler_yolo
```

Output:

```text
reports/external/nestler_yolo_audit.json
reports/external/nestler_yolo_image_stats.csv
```

## 4. Download dataset Roboflow Tier 1

Ambil API key gratis:

```text
https://app.roboflow.com/settings/api
```

Set environment variable:

```powershell
$env:ROBOFLOW_API_KEY="YOUR_KEY"
```

Install dependency jika belum:

```powershell
pip install roboflow
```

Download semua kandidat:

```powershell
python scripts/download_roboflow_datasets.py
```

Atau satu dataset:

```powershell
python scripts/download_roboflow_datasets.py --only chicken_detection_fum
```

## 5. Validasi dataset Roboflow hasil download

Contoh:

```powershell
python scripts/validate_external_dataset.py --name chicken_detection_fum --root data/external/chicken_detection_fum
python scripts/validate_external_dataset.py --name chicken_count --root data/external/chicken_count
python scripts/validate_external_dataset.py --name broiler_healthy_sick --root data/external/broiler_healthy_sick
```

## 6. Mode riset yang sudah diputuskan

Untuk dataset tanpa umur/week:

```text
Mode anomaly-relatif
```

Artinya:
- Tidak memakai Cobb500 absolut
- Tidak mengklaim estimasi berat gram aktual
- Fokus ke:
  - distribusi ukuran bbox per image
  - koreksi radial / depth-light
  - anomaly relatif terhadap median image/dataset
  - domain shift antar dataset

## 7. Checkpoint file

Update progres di:

```text
CHECKPOINT.md
```

Status saat ini:

```text
FASE 1.3 — IN PROGRESS
PIO baseline tervalidasi
NESTLER downloaded
Script validasi & konversi siap
Roboflow menunggu API key
```
