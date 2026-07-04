# Dokumentasi Project: Estimasi Berat dan Deteksi Anomali Broiler Berbasis YOLO + Cobb500

## 1. Ringkasan Project

Project ini membuat pipeline analisis berat ayam broiler dari dataset gambar dan label YOLO. Tujuan akhirnya adalah:

1. Membaca dataset gambar ayam dan label bounding box YOLO.
2. Mengekstrak fitur ukuran ayam dari bounding box.
3. Mengestimasi berat relatif ayam berdasarkan umur minggu dan standar performa Cobb500.
4. Mengurangi bias kamera memakai koreksi radial dan perspective-light.
5. Menghitung anomaly pada beberapa level:
   - bbox/global week-house,
   - image-level flock condition,
   - individual anomaly berbasis konteks image,
   - percentile anomaly sesuai paper threshold selection.
6. Menghasilkan laporan CSV, JSON, dan HTML.

Pipeline final memakai **Cobb500 As-Hatched** sebagai standar berat. Ross 308 sempat diuji, tetapi sudah dihapus dari pipeline agar standar tidak bercampur.

---

## 2. Struktur Dataset

Input utama:

```text
data/images/train/*.jpg
data/images/val/*.jpg
data/labels/train/*.txt
data/labels/val/*.txt
data/FilePrefixCode.xlsx
data/dataset.yaml
data/classes.txt
```

Format label YOLO:

```text
class_id x_center_norm y_center_norm width_norm height_norm
```

Mapping umur dari nama file:

```text
C-W1-XXXX -> Commercial, Week 1, age_days = 7
P-W1-XXXX -> Prototype, Week 1, age_days = 7
C-W2-XXXX -> Commercial, Week 2, age_days = 14
...
P-W6-XXXX -> Prototype, Week 6, age_days = 42
```

File yang tidak mengikuti pola `C-Wx-*` atau `P-Wx-*` di-skip dari estimasi Cobb500 karena umur minggu tidak diketahui.

---

## 3. Script yang Dibuat

### 3.1 `scripts/common.py`

Fungsi umum pipeline:

- konstanta path project,
- referensi berat Cobb500,
- parser file XLSX ringan,
- parser metadata filename,
- pembaca label YOLO,
- pembaca dimensi image JPEG/PNG tanpa library eksternal,
- fungsi statistik: `mean`, `median`, `stdev`, `percentile`,
- writer CSV/JSON.

Referensi berat final:

```text
COBB500_AS_HATCHED
```

Ross 308 sudah dihapus dari source pipeline.

### 3.2 `scripts/audit_dataset.py`

Mengaudit dataset:

- jumlah image dan label per split,
- label tanpa image,
- image tanpa label,
- jumlah bbox per image,
- image size,
- distribusi week dan house,
- bbox invalid,
- export konfigurasi Cobb500.

Output:

```text
configs/prefix_mapping.csv
configs/cobb500_as_hatched.csv
reports/dataset_audit.json
```

### 3.3 `scripts/extract_bbox_features.py`

Mengekstrak fitur per bounding box.

Output:

```text
features/bbox_features.csv
features/bbox_feature_skips.csv
```

Fitur yang dihitung:

```text
width_px
height_px
minor_axis
major_axis
ellipse_area
center_x_px
center_y_px
bottom_y_norm
radius_from_center_px
radius_norm
```

Bbox invalid di-skip jika:

```text
class_id != 0
x/y di luar 0..1
width_norm <= 0
height_norm <= 0
```

### 3.4 `scripts/estimate_weight_anomalies.py`

Estimasi berat awal berbasis Cobb500 dan median group.

Output:

```text
features/weight_estimates.csv
reports/anomalies_individual.csv
reports/anomalies_by_week.csv
reports/anomaly_summary.json
reports/anomaly_report.html
reports/plots/cobb_vs_estimated.svg
reports/overlays/*.svg
```

### 3.5 `scripts/compare_camera_corrections.py`

Membandingkan beberapa model baseline:

```text
original_median
original_mean
radial_median
radial_mean
radial_depth_median
radial_depth_mean
```

Output:

```text
features/weight_estimates_compare.csv
reports/anomaly_baseline_comparison.csv
reports/anomaly_baseline_comparison.html
reports/anomaly_baseline_comparison.json
reports/anomalies_consensus.csv
reports/correction_factors.json
```

Model final yang dipakai untuk laporan utama:

```text
radial_depth_median
```

### 3.6 `scripts/xue_light_calibration.py`

Diagnostic opsional berbasis paper Xue CVPR 2019. Tujuan script ini bukan melakukan rectification penuh, tetapi mengecek apakah dataset punya dukungan garis panjang untuk pendekatan plumb-line/fisheye straight-line calibration.

Jika `opencv-python` tersedia:

```text
Canny edge detection
HoughLinesP
long line segment counting
```

Jika `opencv-python` tidak tersedia, pipeline tetap lanjut dan menulis:

```text
status: opencv_unavailable
```

Output:

```text
configs/xue_light_calibration.json
reports/xue_light_calibration.json
```

Status terakhir setelah instalasi `opencv-python`:

```text
status = completed
sampled_images = 20
total_line_segments = 27,852
total_long_line_segments = 22,074
long_line_support_per_image = 1,103.7
weighted_mean_abs_angle_deg = 32.44
recommendation = Feasible for Xue/plumb-line calibration
```

Interpretasi: dataset memiliki dukungan garis panjang yang sangat kuat untuk diagnostic Xue/plumb-line. Ini belum berarti fisheye rectification penuh sudah dilakukan; hasil ini menunjukkan bahwa pendekatan kalibrasi garis lurus layak dikembangkan sebagai tahap berikutnya.

### 3.7 `scripts/image_level_anomaly.py`

Script final untuk anomaly berbasis image context dan percentile paper.

Input:

```text
features/weight_estimates_compare.csv
```

Output:

```text
features/weight_estimates_image_context.csv
reports/image_level_anomalies.csv
reports/image_level_anomaly_summary.json
reports/image_level_anomaly_report.html
reports/final_individual_anomaly_candidates.csv
reports/final_individual_critical_anomalies.csv
reports/percentile_paper_individual_anomalies.csv
reports/percentile_paper_critical_anomalies.csv
reports/percentile_paper_anomaly_report.html
```

### 3.8 `scripts/run_pipeline.py`

Runner utama pipeline:

```text
audit_dataset.py
extract_bbox_features.py
estimate_weight_anomalies.py
compare_camera_corrections.py
xue_light_calibration.py
image_level_anomaly.py
```

Command:

```powershell
python scripts/run_pipeline.py
```

---

## 4. Paper Referensi yang Akhirnya Dipakai

### 4.1 Cobb500 Broiler Performance & Nutrition Supplement 2022

File:

```text
2022-Cobb500-Broiler-Performance-Nutrition-Supplement_copy.pdf
```

Peran:

- standar target berat broiler,
- anchor estimasi gram per minggu,
- dasar perbandingan flock/image terhadap standar performa.

Target Cobb500 As-Hatched yang dipakai:

```text
W1 / day 7  = 202 g
W2 / day 14 = 570 g
W3 / day 21 = 1116 g
W4 / day 28 = 1783 g
W5 / day 35 = 2521 g
W6 / day 42 = 3278 g
```

### 4.2 Automated precision weighing: Leveraging 2D video feature analysis and machine learning for live body weight estimation of broiler chickens

File:

```text
1-s2.0-S2772375525000279-main_copy.pdf
```

Peran:

- dasar bahwa fitur 2D bisa dipakai untuk estimasi berat broiler,
- mendukung penggunaan `minor_axis` dan umur/age sebagai fitur penting,
- paper menunjukkan fitur terbaik adalah kombinasi **minor ellipse axis + age**,
- paper melaporkan mean relative error sekitar `7.0 ± 5.8%` pada eksperimen besar.

Implikasi ke project:

- pipeline memakai `minor_axis` sebagai fitur utama,
- umur diambil dari week filename,
- estimasi tanpa data timbang manual tetap dianggap estimasi relatif, bukan model regresi final.

### 4.3 DaFIR: Distortion-Aware Representation Learning for Fisheye Image Rectification

File:

```text
DaFIR_Distortion-Aware_Representation_Learning_for_Fisheye_Image_Rectification.pdf
```

Peran:

- dasar koreksi radial/fisheye,
- paper menyatakan tingkat distorsi fisheye berkaitan dengan jarak pixel dari pusat gambar,
- distorsi bersifat radially symmetric.

Implikasi ke project:

```text
radius_norm = distance(bbox_center, image_center) / max_image_radius
```

Lalu dibuat koreksi radial berdasarkan bin radius.

### 4.4 An End-to-End Depth-Based Pipeline for Selfie Image Rectification

File:

```text
An_End-to-End_Depth-Based_Pipeline_for_Selfie_Image_Rectification.pdf
```

Peran:

- dasar ide bahwa ukuran visual perlu dinormalisasi terhadap perspektif/depth,
- paper memakai depth untuk unproject 2D ke 3D dan reproject ke virtual camera yang lebih jauh.

Implikasi ke project:

- pipeline menambahkan `bottom_y_norm` sebagai proxy depth/perspective-light,
- ayam lebih bawah gambar diasumsikan cenderung lebih dekat kamera.

### 4.5 Extending Foundational Monocular Depth Estimators to Fisheye Cameras with Calibration Tokens

File:

```text
Gangopadhyay_Extending_Foundational_Monocular_Depth_Estimators_to_Fisheye_Cameras_with_Calibration_ICCV_2025_paper.pdf
```

Peran:

- catatan bahwa depth estimator biasa bias pada fisheye tanpa kalibrasi,
- memperkuat alasan bahwa correction harus sadar intrinsic/distortion camera.

Implikasi ke project:

- `bottom_y_norm` hanya dianggap pendekatan ringan,
- depth sebenarnya/homography/fisheye-aware depth dibutuhkan untuk akurasi lebih tinggi.

### 4.6 Learning to Calibrate Straight Lines for Fisheye Image Rectification

File:

```text
Xue_Learning_to_Calibrate_Straight_Lines_for_Fisheye_Image_Rectification_CVPR_2019_paper.pdf
```

Peran:

- dasar diagnostic Xue-light/plumb-line calibration,
- paper memakai asumsi garis lurus di dunia nyata harus menjadi lurus setelah rectification,
- cocok jika image kandang memiliki feeder, pipa, dinding, atau garis lantai yang jelas.

Implikasi ke project:

- dibuat script `xue_light_calibration.py`,
- script mengecek dukungan garis panjang dengan Canny + HoughLinesP jika OpenCV tersedia,
- belum melakukan full fisheye rectification.

### 4.7 Comparing Threshold Selection Methods for Network Anomaly Detection

File:

```text
Comparing_Threshold_Selection_Methods_for_Network_Anomaly_Detection.pdf
```

Peran:

- dasar metode threshold percentile,
- paper mendefinisikan percentile method:

```text
T = perc(k, X)
```

- paper menguji `k = 97, 98, 99`, dan konfigurasi statistics-based terbaik yang ditampilkan memakai `k = 99`.

Implikasi ke project:

- warning memakai `P97`,
- critical memakai `P99`,
- anomaly score dibuat satu arah:

```text
X = abs(log(estimated_weight / image_median_weight))
```

---

## 5. Paper yang Dipertimbangkan tetapi Tidak Dipakai sebagai Standar Utama

### Ross 308 Performance Objectives 2022

File:

```text
RossxRoss308-BroilerPerformanceObjectives2022-EN_copy.pdf
```

Status:

- sempat dipakai untuk mode `Ross-calibrated`,
- kemudian dihapus dari pipeline,
- keputusan final: gunakan Cobb500 saja.

Alasan:

- metadata dataset tidak menyebut strain ayam,
- pipeline awal dan standar utama memakai Cobb500,
- memakai dua standar sekaligus membuat interpretasi laporan membingungkan,
- user memutuskan tetap pakai Cobb500.

File output Ross yang sempat dibuat sudah dihapus. Ross tersisa hanya sebagai PDF sumber.

---

## 6. Konfigurasi Final Cobb500

File konfigurasi:

```text
configs/cobb500_as_hatched.csv
```

Mapping week:

```text
week 1 -> day 7
week 2 -> day 14
week 3 -> day 21
week 4 -> day 28
week 5 -> day 35
week 6 -> day 42
```

Target mingguan:

```text
W1 = 202 g
W2 = 570 g
W3 = 1116 g
W4 = 1783 g
W5 = 2521 g
W6 = 3278 g
```

---

## 7. Ekstraksi Fitur Bounding Box

Untuk setiap bbox YOLO:

```text
width_px  = w_norm * image_width
height_px = h_norm * image_height
minor_axis = min(width_px, height_px)
major_axis = max(width_px, height_px)
ellipse_area = pi * minor_axis * major_axis
```

Koordinat bbox:

```text
center_x_px = x_center_norm * image_width
center_y_px = y_center_norm * image_height
x1 = center_x_px - width_px / 2
y1 = center_y_px - height_px / 2
x2 = center_x_px + width_px / 2
y2 = center_y_px + height_px / 2
```

Fitur posisi kamera:

```text
bottom_y_norm = y2 / image_height
radius_from_center_px = sqrt((center_x_px - image_width/2)^2 + (center_y_px - image_height/2)^2)
radius_norm = radius_from_center_px / sqrt((image_width/2)^2 + (image_height/2)^2)
```

---

## 8. Estimasi Berat Awal Cobb500

Grouping baseline:

```text
group_key = house + week
contoh: Commercial_W1, Prototype_W3
```

Baseline visual per group:

```text
group_median_minor_axis
group_median_ellipse_area
```

Rasio visual:

```text
minor_ratio = minor_axis / group_median_minor_axis
area_ratio = ellipse_area / group_median_ellipse_area
```

Estimasi berat:

```text
est_minor = cobb_weight_g * minor_ratio
est_area = cobb_weight_g * sqrt(area_ratio)
estimated_weight_g = 0.7 * est_minor + 0.3 * est_area
```

Alasan bobot:

```text
70% minor_axis
30% sqrt(area)
```

Minor axis diprioritaskan karena paper broiler weight menunjukkan minor axis + age sebagai fitur kuat.

---

## 9. Koreksi Kamera: Mean/Median, Radial, Radial-Depth

Script:

```text
scripts/compare_camera_corrections.py
```

Model pembanding:

```text
original_median
original_mean
radial_median
radial_mean
radial_depth_median
radial_depth_mean
```

### 9.1 Original

Menggunakan fitur mentah:

```text
minor_axis
ellipse_area
```

### 9.2 Radial correction

Menggunakan bin `radius_norm`.

Konfigurasi:

```text
BIN_COUNT = 6
```

Faktor radial dihitung dari rasio median bbox-size terhadap median group:

```text
raw_group_median_minor = median(minor_axis per house-week)
ratio = minor_axis / raw_group_median_minor
radial_scale_factor = median(ratio per radius_bin)
```

Clamping:

```text
min factor = 0.55
max factor = 1.85
```

Koreksi:

```text
radial_corrected_minor_axis = minor_axis / radial_scale_factor
radial_corrected_ellipse_area = ellipse_area / radial_scale_factor^2
```

### 9.3 Perspective-light / depth-light correction

Menggunakan bin `bottom_y_norm` sebagai proxy depth.

```text
perspective_scale_factor = median(radial_corrected_minor_axis / group_median_radial_minor per bottom_y_bin)
```

Koreksi:

```text
radial_depth_corrected_minor_axis = radial_corrected_minor_axis / perspective_scale_factor
radial_depth_corrected_ellipse_area = radial_corrected_ellipse_area / perspective_scale_factor^2
```

Model final:

```text
radial_depth_median
```

Kolom berat final untuk anomaly utama:

```text
radial_depth_median_estimated_weight_g
```

---

## 10. Hasil Perbandingan Konfigurasi Kamera

Total bbox valid:

```text
321,427
```

Hasil anomaly global bbox:

```text
original_median       244,746 anomaly = 76.14%
original_mean         243,948 anomaly = 75.90%
radial_median         187,651 anomaly = 58.38%
radial_mean           187,359 anomaly = 58.29%
radial_depth_median   187,135 anomaly = 58.22%
radial_depth_mean     187,244 anomaly = 58.25%
```

Consensus:

```text
all_models_anomaly = 137,982
all_models_normal  = 30,092
mixed              = 153,353
```

Interpretasi:

- mean vs median tidak banyak berbeda,
- koreksi radial memberi penurunan besar dari 76% ke 58%,
- depth-light memberi tambahan kecil,
- anomaly global bbox masih terlalu tinggi jika dipakai sebagai angka final.

---

## 11. Image-Level Anomaly

Tujuan:

- mengurangi false positive dari bbox individual,
- memanfaatkan fakta bahwa satu image berisi banyak ayam dengan bias kamera yang mirip,
- mengevaluasi kondisi flock/gambar terhadap Cobb500.

Untuk setiap image:

```text
image_mean_weight_g = mean(radial_depth_median_estimated_weight_g)
image_median_weight_g = median(radial_depth_median_estimated_weight_g)
image_std_weight_g = stdev(radial_depth_median_estimated_weight_g)
image_mad_weight_g = median(abs(weight - image_median_weight_g))
image_cv_pct = image_std_weight_g / image_mean_weight_g * 100
image_cobb_diff_pct = (image_mean_weight_g - cobb_weight_g) / cobb_weight_g * 100
```

Threshold image:

```text
warning_image_below_cobb  jika image_cobb_diff_pct < -10%
critical_image_below_cobb jika image_cobb_diff_pct < -20%
warning_image_above_cobb  jika image_cobb_diff_pct > +10%
critical_image_above_cobb jika image_cobb_diff_pct > +20%
warning_uniformity_problem_cv_gt_20  jika image_cv_pct > 20%
critical_uniformity_problem_cv_gt_30 jika image_cv_pct > 30%
low_sample_count jika count < 20
normal_image jika tidak kena flag
```

Output:

```text
reports/image_level_anomalies.csv
reports/image_level_anomaly_report.html
reports/image_level_anomaly_summary.json
```

Hasil terakhir:

```text
total_images = 1,468
abnormal_images = 617
abnormal_image_rate_pct = 42.03%
```

---

## 12. Individual Anomaly Berbasis Image Context: MAD / Robust Z

Untuk setiap bbox:

```text
relative_to_image_median = estimated_weight / image_median_weight
robust_z_image = 0.6745 * (estimated_weight - image_median_weight) / image_mad_weight
```

Threshold individual manual/MAD:

```text
warning_low_vs_image jika relative_to_image_median < 0.75 atau robust_z_image < -3.5
critical_low_vs_image jika relative_to_image_median < 0.65 atau robust_z_image < -4.5

warning_high_vs_image jika relative_to_image_median > 1.25 atau robust_z_image > 3.5
critical_high_vs_image jika relative_to_image_median > 1.35 atau robust_z_image > 4.5

camera_corrected_model_anomaly jika radial_depth_median_is_anomaly = True
from_critical_image jika image asal critical
```

Final level:

```text
normal   = tidak ada flag
warning  = ada warning atau camera_corrected_model_anomaly
critical = ada critical
```

Output:

```text
features/weight_estimates_image_context.csv
reports/final_individual_anomaly_candidates.csv
reports/final_individual_critical_anomalies.csv
```

Hasil terakhir:

```text
final_candidate_bboxes = 188,359 = 58.60%
critical_bboxes = 32,099 = 9.99%
```

Catatan:

- angka candidate masih besar karena memasukkan warning dan camera-corrected model anomaly,
- angka critical 9.99% lebih layak untuk anomaly kuat dibanding 58.60%.

---

## 13. Percentile Threshold sesuai Paper

Dasar paper:

```text
Comparing Threshold Selection Methods for Network Anomaly Detection
```

Metode:

```text
T = perc(k, X)
```

Konfigurasi paper yang diadaptasi:

```text
k = 97 untuk warning
k = 99 untuk critical
```

Karena anomaly berat ayam bisa dua arah, dibuat anomaly score satu arah:

```text
paper_percentile_score = abs(log(radial_depth_median_estimated_weight_g / image_median_weight_g))
```

Interpretasi score:

```text
score kecil = berat dekat median image
score besar = berat jauh dari median image
score berlaku untuk ayam terlalu kecil maupun terlalu besar
```

Threshold:

```text
paper_percentile_threshold_p97 = percentile(score, 0.97)
paper_percentile_threshold_p99 = percentile(score, 0.99)
```

Level:

```text
normal   jika score < P97
warning  jika P97 <= score < P99
critical jika score >= P99
```

Fallback context:

```text
Jika bbox_count_image >= 100:
  P97/P99 dihitung per image

Jika bbox_count_image < 100:
  P97/P99 dihitung per house-week
```

Konfigurasi:

```text
MIN_IMAGE_PERCENTILE_COUNT = 100
```

Output:

```text
reports/percentile_paper_individual_anomalies.csv
reports/percentile_paper_critical_anomalies.csv
reports/percentile_paper_anomaly_report.html
```

Hasil terakhir:

```text
paper_percentile_candidate_bboxes = 10,085 = 3.14%
paper_percentile_critical_bboxes = 3,667 = 1.14%
```

Context split:

```text
image context = 287,536 bbox
house_week fallback = 33,891 bbox
```

Interpretasi:

- P97+ adalah top 3% paling ekstrem dalam konteksnya,
- P99+ adalah top 1% paling ekstrem dalam konteksnya,
- ini metode paling konservatif dan paling cocok untuk mengurangi false positive.

---

## 14. Xue-Light Calibration Diagnostic

Script:

```text
scripts/xue_light_calibration.py
```

Tujuan:

- mengecek apakah dataset memungkinkan pendekatan straight-line calibration ala Xue CVPR 2019,
- bukan full rectification.

Jika OpenCV tersedia:

```text
gray image
GaussianBlur
Canny edge
HoughLinesP
count long lines >= 250 px
```

Output:

```text
reports/xue_light_calibration.json
configs/xue_light_calibration.json
```

Status terakhir setelah instalasi OpenCV:

```json
{
  "status": "completed",
  "sampled_images": 20,
  "total_line_segments": 27852,
  "total_long_line_segments": 22074,
  "long_line_support_per_image": 1103.7,
  "weighted_mean_abs_angle_deg": 32.44,
  "recommendation": "Feasible for Xue/plumb-line calibration"
}
```

Implikasi:

- OpenCV sudah tersedia,
- Xue-light diagnostic berhasil berjalan,
- banyak long line terdeteksi pada sampel image,
- pendekatan Xue/plumb-line feasible untuk tahap lanjutan,
- tahap saat ini masih diagnostic, belum menghasilkan parameter undistortion `k1/k2/k3` atau transformasi bbox.

---

## 15. Output Final dan Kegunaannya

### 15.1 Audit dataset

```text
reports/dataset_audit.json
```

Gunakan untuk cek kualitas dataset, image count, bbox invalid, filename tanpa mapping umur.

### 15.2 Fitur bbox

```text
features/bbox_features.csv
```

Berisi fitur mentah per bbox.

### 15.3 Estimasi berat Cobb500 awal

```text
features/weight_estimates.csv
reports/anomaly_report.html
reports/anomalies_individual.csv
reports/anomalies_by_week.csv
```

Laporan awal. Tidak direkomendasikan sebagai angka final karena anomaly bbox global terlalu sensitif.

### 15.4 Perbandingan konfigurasi kamera

```text
features/weight_estimates_compare.csv
reports/anomaly_baseline_comparison.html
reports/anomaly_baseline_comparison.csv
reports/anomaly_baseline_comparison.json
reports/anomalies_consensus.csv
reports/correction_factors.json
```

Gunakan untuk membandingkan original, radial, radial-depth, mean, median.

### 15.5 Image-level anomaly

```text
reports/image_level_anomalies.csv
reports/image_level_anomaly_report.html
reports/image_level_anomaly_summary.json
```

Gunakan untuk monitoring kondisi per image/flock.

### 15.6 Individual anomaly manual/MAD

```text
reports/final_individual_anomaly_candidates.csv
reports/final_individual_critical_anomalies.csv
```

Gunakan jika ingin threshold berbasis relative-to-image-median dan robust z.

### 15.7 Individual anomaly percentile paper

```text
reports/percentile_paper_individual_anomalies.csv
reports/percentile_paper_critical_anomalies.csv
reports/percentile_paper_anomaly_report.html
```

Rekomendasi final untuk anomaly individual konservatif.

---

## 16. Hasil Angka Terakhir

Dataset valid:

```text
Total bbox valid = 321,427
Total image = 1,468
```

Image-level:

```text
Abnormal image = 617
Abnormal image rate = 42.03%
```

Global bbox model comparison:

```text
original_median anomaly = 76.14%
original_mean anomaly = 75.90%
radial_median anomaly = 58.38%
radial_mean anomaly = 58.29%
radial_depth_median anomaly = 58.22%
radial_depth_mean anomaly = 58.25%
```

MAD/manual image-context:

```text
final candidate bbox = 188,359 = 58.60%
critical bbox = 32,099 = 9.99%
```

Percentile paper:

```text
P97+ candidate bbox = 10,085 = 3.14%
P99+ critical bbox = 3,667 = 1.14%
```

Xue-light:

```text
status = completed
sampled_images = 20
total_line_segments = 27,852
total_long_line_segments = 22,074
long_line_support_per_image = 1,103.7
recommendation = Feasible for Xue/plumb-line calibration
```

---

## 17. Rekomendasi Pemakaian Hasil

### Untuk laporan utama individual anomaly

Pakai:

```text
reports/percentile_paper_critical_anomalies.csv
```

Alasan:

- paling konservatif,
- mengikuti metode percentile threshold dari paper,
- hanya top 1% paling ekstrem per image/house-week context,
- mengurangi false positive.

### Untuk kandidat review luas

Pakai:

```text
reports/percentile_paper_individual_anomalies.csv
```

Alasan:

- berisi P97+,
- top 3% paling ekstrem,
- cocok untuk inspeksi manual lebih luas.

### Untuk monitoring per gambar/flock

Pakai:

```text
reports/image_level_anomalies.csv
```

Alasan:

- membandingkan rata-rata image ke Cobb500,
- memberi uniformity/CV per image,
- lebih stabil daripada bbox individual.

### Untuk analisis bias kamera

Pakai:

```text
reports/anomaly_baseline_comparison.html
reports/correction_factors.json
```

Alasan:

- menunjukkan efek original vs radial vs radial-depth,
- membantu melihat apakah anomaly dipengaruhi posisi kamera.

---

## 18. Keterbatasan Validitas

1. Belum ada data timbang manual per ayam.

   Estimasi ini relatif terhadap Cobb500 dan fitur visual, bukan regresi aktual terkalibrasi.

2. Tidak ada tracking ayam antar frame.

   Setiap bbox dianggap observasi individu. Jika video frame berurutan, satu ayam bisa muncul berkali-kali.

3. Koreksi radial dan depth masih pendekatan ringan.

   Tidak ada kalibrasi kamera fisik, homography lantai, atau depth map nyata.

4. Xue-light belum aktif karena OpenCV tidak tersedia.

   Full straight-line calibration belum dilakukan.

5. Cobb500 dipakai karena keputusan final project.

   Jika strain aktual bukan Cobb500, interpretasi terhadap standar performa harus disesuaikan.

6. Percentile anomaly adalah anomaly relatif dalam konteks image/house-week.

   P99 berarti top 1% paling ekstrem, bukan otomatis sakit atau berat aktual salah.

---

## 19. Langkah Pengembangan Berikutnya

Prioritas tinggi:

1. Tambahkan data berat aktual manual untuk kalibrasi regresi.
2. Tambahkan tracking ayam antar frame agar satu ayam tidak dihitung berulang.
3. Lanjutkan Xue-light dari diagnostic ke estimasi parameter radial distortion.
4. Jika parameter radial stabil, transformasi bbox dengan correction map/undistortion.
5. Tambahkan homography lantai untuk koreksi perspektif lebih baik.

Prioritas menengah:

1. Tambahkan model regresi ringan:

```text
weight ~ minor_axis + age + radius_norm + bottom_y_norm
```

2. Bandingkan Linear, Robust Linear, Random Forest, Gradient Boosting jika data timbang tersedia.
3. Tambahkan aggregation per image/video/session.
4. Tambahkan dashboard visual.

Prioritas rendah:

1. Integrasi depth estimator fisheye-aware.
2. Full DaFIR-style rectification.
3. Full Xue-style learned calibration.

---

## 20. Kesimpulan Final

Pipeline final sekarang memakai:

```text
Standar berat: Cobb500 As-Hatched
Fitur utama: minor_axis + ellipse_area + age/week
Koreksi kamera: radial_depth_median
Image anomaly: Cobb diff + CV
Individual anomaly konservatif: percentile method P97/P99
Final critical recommended: percentile P99
```

Angka final yang paling aman dipakai:

```text
Percentile P99 critical anomaly = 3,667 bbox = 1.14%
```

File final paling penting:

```text
reports/percentile_paper_critical_anomalies.csv
reports/percentile_paper_anomaly_report.html
reports/image_level_anomalies.csv
reports/image_level_anomaly_report.html
```
