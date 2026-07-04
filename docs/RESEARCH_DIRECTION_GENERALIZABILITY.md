# Arah Riset: Generalizability Pipeline Estimasi/Anomali Broiler pada Multi-Dataset

Tanggal: 2026-06-27

## 1. Masalah Riset

Dataset utama proyek ini bersumber dari paper PIO (*PIO, A Large-Scale Dataset for Broiler Chicken Detection under Real Poultry Farming Conditions*). Dataset tersebut sudah tersedia di Zenodo dan telah digunakan sebagai baseline. Namun, untuk memperkuat arah riset, dosen meminta pengujian apakah modeling/pipeline proyek ini dapat diterapkan pada beberapa dataset lain sehingga tidak hanya valid pada satu kondisi kandang.

Kendala utama:

1. Beberapa dataset dalam Table 1 paper PIO tidak tersedia publik atau hanya tersedia by request.
2. Dataset eksternal yang publik umumnya tidak memiliki:
   - umur ayam (`week` / `age_days`),
   - strain ayam,
   - ground-truth berat aktual,
   - metadata kandang yang setara dengan PIO.
3. Karena itu, klaim berat absolut berbasis Cobb500 tidak boleh dipaksakan pada dataset eksternal.

Solusi metodologis:

- **PIO** tetap dipakai untuk mode absolut: fitur bbox + metadata week + Cobb500.
- **Dataset eksternal** dipakai untuk mode relatif: visual relative-size anomaly tanpa klaim gram aktual.

---

## 2. Tujuan Riset yang Disarankan

Tujuan utama:

> Menguji apakah pipeline visual-anomaly berbasis fitur bounding box, koreksi kamera ringan, dan threshold percentile tetap stabil saat diterapkan pada beberapa dataset ayam/broiler dengan kondisi visual berbeda.

Tujuan turunan:

1. Menguji apakah fitur visual `minor_axis`, `ellipse_area`, `radius_norm`, dan `bottom_y_norm` dapat diekstrak konsisten pada dataset publik lain.
2. Menguji apakah koreksi radial/depth-light mengurangi variasi ukuran visual lintas posisi kamera.
3. Menguji apakah threshold percentile P97/P99 menghasilkan kandidat anomali yang terkendali di berbagai dataset.
4. Mengidentifikasi dataset mana yang layak untuk klaim generalizability dan mana yang hanya berfungsi sebagai robustness check.

---

## 3. Dataset yang Digunakan

### 3.1 Baseline utama

| Dataset | Sumber | Format | Images | BBox | Kegunaan |
|---|---|---|---:|---:|---|
| PIO | Zenodo / Scientific Data 2026 | YOLO bbox | 1,487 | 327,283 valid | Baseline utama; punya week metadata, cocok untuk Cobb500 absolute mode |

### 3.2 Dataset eksternal untuk generalizability

| Dataset | Sumber | Format awal | Format final | Images | BBox | Kegunaan |
|---|---|---|---|---:|---:|---|
| NESTLER Poultry Behaviour | Zenodo | video + JSON bbox/keypoint | YOLO bbox | 480 sampled | 4,043 | Sparse/medium density; domain berbeda dari PIO |
| Broiler Healthy & Sick | Roboflow | YOLO segmentation | YOLO bbox | 491 | 491 | Dataset sparse; cocok sebagai robustness check, bukan flock-level density |
| Broiler Instance Segmentation | Roboflow | YOLO segmentation | YOLO bbox | 200 | 10,570 | Broiler dense; cocok untuk generalizability |
| Chicken Count | Roboflow | YOLO bbox | YOLO bbox | 178 | 3,646 | Counting + mixed resolution; domain shift tinggi |
| FUM Chicken Detection | Roboflow | YOLO bbox | YOLO bbox | 326 | 29,355 | Dense detection; cocok untuk generalizability high-density |

Dataset yang tidak dipakai sebagai dataset utama:

- Chicks4FreeID: publik, tetapi cropped image + identity label; tidak ada bounding box scene.
- ChickTrack, Dense-Chicken, Zhuang & Zhang: tidak ditemukan sebagai dataset publik siap-download.
- Shams 2025 YOLO-seg weight estimation: sangat relevan secara topik, tetapi data tidak publik; hanya available from corresponding author.

---

## 4. Dua Mode Analisis

### 4.1 Mode absolut — hanya untuk PIO

Dipakai ketika dataset punya metadata umur/week.

```text
estimated_weight_g = fungsi(minor_axis, ellipse_area, age/week, Cobb500)
```

Output penting:

```text
reports/percentile_paper_critical_anomalies.csv
reports/image_level_anomalies.csv
reports/anomaly_baseline_comparison.html
```

Klaim yang aman:

> Pada dataset PIO, pipeline dapat menghasilkan estimasi berat relatif berbasis Cobb500 karena filename menyediakan metadata week dan dataset memiliki konteks growth stage.

### 4.2 Mode relatif — untuk dataset eksternal

Dipakai ketika dataset tidak punya umur/week.

```text
relative_to_image_median = corrected_visual_size / median_visual_size_in_context
relative_percentile_score = abs(log(relative_to_image_median))
```

Output penting:

```text
reports/external/cross_dataset_relative_summary.csv
reports/external/cross_dataset_relative_report.html
```

Klaim yang aman:

> Pada dataset eksternal, pipeline tidak mengestimasi berat gram aktual. Pipeline hanya mengukur deviasi ukuran visual relatif terhadap konteks image/dataset.

---

## 5. Hasil Awal Generalizability

| Dataset | BBox | Median bbox/image | P97+ candidate | P99+ critical | Median image CV | Interpretasi |
|---|---:|---:|---:|---:|---:|---|
| NESTLER | 4,043 | 9.0 | 3.02% | 1.01% | 50.15 | Threshold stabil, tetapi density rendah/medium |
| Broiler Healthy & Sick | 491 | 1.0 | 0.00% | 0.00% | 0.00 | Terlalu sparse untuk image-context anomaly; hanya robustness check |
| Broiler Instance Segmentation | 10,570 | 53.0 | 3.78% | 1.89% | 22.96 | Dataset broiler dense; kuat untuk klaim generalizability relatif |
| Chicken Count | 3,646 | 18.0 | 3.73% | 2.58% | 22.82 | Mixed resolution; domain shift tinggi, P99 lebih besar |
| FUM Chicken Detection | 29,355 | 88.5 | 3.52% | 1.48% | 34.82 | Dense; kuat untuk pengujian high-density |

Interpretasi:

1. P97+ umumnya berada sekitar 3–4%, mendekati target metode percentile.
2. P99+ umumnya berada sekitar 1–2%, kecuali dataset mixed resolution (`chicken_count`) yang lebih bervariasi.
3. Dataset dense (`broiler_instance_seg`, `chicken_detection_fum`) paling cocok untuk argumen generalizability.
4. Dataset sparse (`broiler_healthy_sick`) tidak cocok untuk menguji flock/image context karena mayoritas hanya satu bbox per image.

---

## 6. Kontribusi Riset yang Bisa Diklaim

### Klaim kuat

1. Pipeline dapat mengekstrak fitur visual bbox secara konsisten pada beberapa dataset publik.
2. Mode relatif memungkinkan evaluasi lintas dataset tanpa memerlukan metadata umur/berat.
3. Threshold percentile P97/P99 menghasilkan kandidat anomali yang relatif terkendali pada dataset dense.
4. Dataset eksternal menunjukkan bahwa pipeline tidak hanya bergantung pada struktur folder PIO.

### Klaim sedang

1. Koreksi radial/depth-light membantu menurunkan sebagian variasi ukuran visual pada beberapa dataset.
2. Dataset dengan resolusi campuran menunjukkan domain shift yang lebih berat dan membutuhkan normalisasi tambahan.

### Klaim yang tidak boleh dibuat

1. Jangan klaim berat aktual berhasil diprediksi pada dataset eksternal.
2. Jangan klaim ayam sakit/kurus secara biologis tanpa ground-truth kesehatan/berat.
3. Jangan klaim model sudah universal untuk semua kandang; hasil masih bergantung pada kualitas bbox, sudut kamera, dan density.

---

## 7. Arah Eksperimen Berikutnya

### Prioritas 1 — Cross-dataset stability analysis

Gunakan hasil Phase 2 untuk membahas:

```text
P97/P99 rate
image CV median
radial/depth correction effect
median bbox per image
dataset density level
```

Tujuan: menunjukkan dataset mana yang stabil dan mana yang domain-shift tinggi.

### Prioritas 2 — Perbaikan normalisasi resolusi dan density

Dataset eksternal memiliki resolusi bervariasi. Tambahkan analisis:

- kelompok resolusi,
- normalized bbox area,
- density per image,
- effect of mixed-resolution on anomaly score.

### Prioritas 3 — Subset dense-only evaluation

Agar klaim generalizability lebih bersih, buat subset:

```text
image dengan bbox_count >= 20
```

Lalu bandingkan P97/P99 hanya pada dataset/image yang cukup dense.

### Prioritas 4 — Jika dataset timbang aktual didapat

Jika nanti koresponden memberi dataset berat aktual:

- gunakan PIO/external relative mode sebagai pretraining/feature validation,
- kalibrasi regresi aktual:

```text
weight ~ minor_axis + ellipse_area + age + radius_norm + bottom_y_norm
```

- bandingkan error dengan paper weight-estimation.

---

## 8. Rekomendasi Narasi untuk Dosen

Versi singkat:

> Karena dataset eksternal publik umumnya tidak menyediakan umur ayam dan data timbang aktual, penelitian diarahkan menjadi dua mode. Dataset PIO tetap dipakai untuk estimasi berat relatif berbasis Cobb500, sedangkan dataset eksternal dipakai untuk menguji generalizability visual-anomaly secara relatif. Hasil awal menunjukkan pipeline dapat berjalan pada beberapa dataset publik dengan density dan resolusi berbeda, dan threshold percentile P97/P99 tetap menghasilkan kandidat anomaly yang terkendali pada dataset dense. Dengan demikian, kontribusi riset diarahkan pada generalisasi metode visual-anomaly dan bukan klaim berat aktual lintas dataset.

Versi teknis:

> The proposed pipeline is evaluated in absolute Cobb500 mode on PIO and in relative anomaly mode on external public poultry datasets. Since external datasets do not provide age or body-weight ground truth, the relative mode uses corrected visual-size features and percentile-based thresholds to quantify within-context deviations. This enables cross-dataset generalizability testing without overclaiming actual weight estimation.

---

## 9. File Pendukung

Pipeline dan laporan:

```text
configs/datasets/external_datasets.json
scripts/extract_external_bbox_features.py
scripts/relative_anomaly_pipeline.py
scripts/run_external_relative_pipeline.py
reports/external/cross_dataset_relative_summary.csv
reports/external/cross_dataset_relative_report.html
PHASE_2_RELATIVE_GENERALIZATION.md
CHECKPOINT.md
```
