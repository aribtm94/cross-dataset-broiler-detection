# Final Best Calibration dan Referensi Project

## 1. Ringkasan Keputusan Final

Project ini adalah pipeline estimasi berat relatif dan deteksi anomali ayam broiler berbasis anotasi YOLO, standar Cobb500, koreksi kamera ringan, dan threshold anomaly konservatif.

Konfigurasi final:

```text
Standar berat final            : Cobb500 As-Hatched
Fitur utama                    : minor_axis + ellipse_area + age/week
Baseline visual                : house-week
Best camera correction         : radial_depth_median
Kolom berat final              : radial_depth_median_estimated_weight_g
Image anomaly                  : Cobb diff + coefficient of variation (CV)
Individual anomaly final       : percentile method P97/P99
Final critical recommended     : percentile P99
```

File output final yang direkomendasikan:

```text
reports/percentile_paper_critical_anomalies.csv
reports/percentile_paper_anomaly_report.html
reports/image_level_anomalies.csv
reports/image_level_anomaly_report.html
```

Angka final paling aman dipakai:

```text
Percentile P99 critical anomaly = 3,667 bbox = 1.14%
```

Alasan: P99 hanya mengambil top 1% observasi paling ekstrem dalam konteks image atau house-week, sehingga lebih konservatif dibanding anomaly bbox global.

---

## 2. Tujuan Calibration

Calibration pada project ini bukan kalibrasi kamera fisik penuh. Calibration dipakai untuk menstabilkan estimasi berat relatif terhadap:

1. umur ayam,
2. perbedaan kandang/house,
3. variasi ukuran visual bbox,
4. bias radial/fisheye dari posisi bbox,
5. bias perspektif dari posisi vertikal bbox,
6. false positive anomaly individual.

Tahap final memakai pendekatan ringan berbasis fitur visual dan statistik robust. Full fisheye undistortion, depth map, homography lantai, dan regresi berbasis berat aktual belum diterapkan.

---

## 3. Dataset dan Input

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

Mapping umur dari filename:

```text
C-W1-XXXX -> Commercial, Week 1, age_days = 7
P-W1-XXXX -> Prototype, Week 1, age_days = 7
C-W2-XXXX -> Commercial, Week 2, age_days = 14
P-W2-XXXX -> Prototype, Week 2, age_days = 14
...
C-W6-XXXX -> Commercial, Week 6, age_days = 42
P-W6-XXXX -> Prototype, Week 6, age_days = 42
```

File yang tidak mengikuti pola `C-Wx-*` atau `P-Wx-*` tidak dipakai untuk estimasi Cobb500 karena umur minggu tidak diketahui.

Dataset valid terakhir:

```text
Total bbox valid = 321,427
Total image      = 1,468
```

---

## 4. Standar Berat Final: Cobb500 As-Hatched

Standar berat final memakai Cobb500 `As-Hatched`.

Target mingguan:

```text
W1 / day 7  = 202 g
W2 / day 14 = 570 g
W3 / day 21 = 1116 g
W4 / day 28 = 1783 g
W5 / day 35 = 2521 g
W6 / day 42 = 3278 g
```

File konfigurasi:

```text
configs/cobb500_as_hatched.csv
```

Ross 308 tidak dipakai sebagai standar final karena metadata dataset tidak memastikan strain ayam dan dua standar membuat interpretasi laporan bercampur.

---

## 5. Ekstraksi Fitur Bbox

Fitur ukuran bbox:

```text
width_px  = width_norm * image_width
height_px = height_norm * image_height
minor_axis  = min(width_px, height_px)
major_axis  = max(width_px, height_px)
ellipse_area = pi * minor_axis * major_axis
```

Fitur posisi kamera:

```text
bottom_y_norm = y2 / image_height
radius_from_center_px = sqrt((center_x_px - image_width/2)^2 + (center_y_px - image_height/2)^2)
radius_norm = radius_from_center_px / sqrt((image_width/2)^2 + (image_height/2)^2)
```

Fitur final yang dipakai:

```text
minor_axis
ellipse_area
age_days / week
radius_norm
bottom_y_norm
```

`minor_axis` diprioritaskan karena referensi estimasi berat broiler berbasis video 2D menunjukkan kombinasi minor ellipse axis dan umur sebagai fitur kuat.

---

## 6. Estimasi Berat Dasar

Baseline visual dihitung per group:

```text
group_key = house + week
contoh: Commercial_W1, Prototype_W3
```

Baseline group:

```text
group_median_minor_axis
group_median_ellipse_area
```

Rasio visual:

```text
minor_ratio = minor_axis / group_median_minor_axis
area_ratio  = ellipse_area / group_median_ellipse_area
```

Formula estimasi:

```text
est_minor = cobb_weight_g * minor_ratio
est_area  = cobb_weight_g * sqrt(area_ratio)
estimated_weight_g = 0.7 * est_minor + 0.3 * est_area
```

Bobot:

```text
70% minor_axis
30% sqrt(ellipse_area)
```

Alasan: `minor_axis` menjadi fitur utama, sedangkan `sqrt(area)` membantu menangkap skala tubuh tanpa membuat area terlalu dominan.

---

## 7. Best Camera Calibration: `radial_depth_median`

Model pembanding:

```text
original_median
original_mean
radial_median
radial_mean
radial_depth_median
radial_depth_mean
```

Model final:

```text
radial_depth_median
```

Alasan pemilihan:

1. Median lebih robust terhadap outlier bbox.
2. Radial correction mengurangi bias posisi dari `radius_norm`.
3. Depth-light correction mengurangi bias perspektif dari `bottom_y_norm`.
4. `radial_depth_median` memberi anomaly rate terbaik di kelompok model median.
5. Model ini dipakai konsisten sebagai input image-level dan percentile anomaly.

Kolom berat final:

```text
radial_depth_median_estimated_weight_g
```

---

## 8. Radial Correction

Radial correction memakai asumsi bahwa distorsi fisheye/radial berkaitan dengan jarak pixel dari pusat gambar.

Konfigurasi:

```text
BIN_COUNT = 6
radial_bin = int(radius_norm * 6)
```

Faktor koreksi:

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

Faktor radial terakhir:

```text
bin 0 = 1.3970442907856695
bin 1 = 1.3250123751856278
bin 2 = 1.200018000270004
bin 3 = 1.0634799447627816
bin 4 = 0.8571292519166362
bin 5 = 0.6500247503712555
```

File:

```text
reports/correction_factors.json
```

---

## 9. Perspective-Light / Depth-Light Correction

Perspective-light memakai `bottom_y_norm` sebagai proxy depth. Ayam lebih bawah pada gambar diasumsikan cenderung lebih dekat kamera, sehingga ukuran visual perlu dinormalisasi.

Faktor:

```text
perspective_scale_factor = median(radial_corrected_minor_axis / group_median_radial_minor per bottom_y_bin)
```

Koreksi:

```text
radial_depth_corrected_minor_axis = radial_corrected_minor_axis / perspective_scale_factor
radial_depth_corrected_ellipse_area = radial_corrected_ellipse_area / perspective_scale_factor^2
```

Faktor perspective terakhir:

```text
bin 0 = 0.9773687461395923
bin 1 = 1.00987551344801
bin 2 = 1.0154614278529475
bin 3 = 1.0
bin 4 = 1.0
bin 5 = 1.0108551628274423
```

Catatan: ini bukan depth map aktual. Ini hanya koreksi ringan berbasis posisi vertikal bbox.

---

## 10. Bukti Perbandingan Model Kamera

Total bbox valid:

```text
321,427
```

Anomaly global bbox per model:

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

1. Mean dan median memberi hasil mirip.
2. Koreksi radial memberi penurunan besar dari sekitar 76% ke sekitar 58%.
3. Depth-light memberi tambahan kecil.
4. `radial_depth_median` dipilih karena robust dan menjadi model final untuk tahap image-context.
5. Anomaly global bbox masih terlalu sensitif, sehingga tidak dijadikan angka final individual anomaly.

File pembanding:

```text
features/weight_estimates_compare.csv
reports/anomaly_baseline_comparison.csv
reports/anomaly_baseline_comparison.html
reports/anomaly_baseline_comparison.json
reports/anomalies_consensus.csv
reports/correction_factors.json
```

---

## 11. Image-Level Anomaly

Image-level dipakai untuk mengurangi false positive karena bbox dalam satu image punya bias kamera yang mirip.

Statistik per image:

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

Hasil terakhir:

```text
total_images = 1,468
abnormal_images = 617
abnormal_image_rate_pct = 42.03%
```

File:

```text
reports/image_level_anomalies.csv
reports/image_level_anomaly_summary.json
reports/image_level_anomaly_report.html
```

---

## 12. Individual Anomaly Manual/MAD

Formula image-context:

```text
relative_to_image_median = estimated_weight / image_median_weight
robust_z_image = 0.6745 * (estimated_weight - image_median_weight) / image_mad_weight
```

Threshold:

```text
warning_low_vs_image jika relative_to_image_median < 0.75 atau robust_z_image < -3.5
critical_low_vs_image jika relative_to_image_median < 0.65 atau robust_z_image < -4.5
warning_high_vs_image jika relative_to_image_median > 1.25 atau robust_z_image > 3.5
critical_high_vs_image jika relative_to_image_median > 1.35 atau robust_z_image > 4.5
```

Hasil terakhir:

```text
final_candidate_bboxes = 188,359 = 58.60%
critical_bboxes        = 32,099  = 9.99%
```

Interpretasi: candidate masih besar karena memasukkan warning dan camera-corrected model anomaly. Angka critical lebih berguna, tetapi masih lebih agresif daripada percentile P99.

File:

```text
reports/final_individual_anomaly_candidates.csv
reports/final_individual_critical_anomalies.csv
```

---

## 13. Final Individual Anomaly: Percentile P97/P99

Metode final paling konservatif memakai percentile threshold.

Formula paper:

```text
T = perc(k, X)
```

Konfigurasi:

```text
k = 97 untuk warning
k = 99 untuk critical
```

Skor anomaly satu arah:

```text
paper_percentile_score = abs(log(radial_depth_median_estimated_weight_g / image_median_weight_g))
```

Interpretasi:

```text
score kecil = berat dekat median image
score besar = berat jauh dari median image
score berlaku untuk ayam terlalu kecil maupun terlalu besar
```

Level:

```text
normal   jika score < P97
warning  jika P97 <= score < P99
critical jika score >= P99
```

Context threshold:

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

Hasil terakhir:

```text
paper_percentile_candidate_bboxes = 10,085 = 3.14%
paper_percentile_critical_bboxes  = 3,667  = 1.14%
```

Context split:

```text
image context       = 287,536 bbox
house_week fallback = 33,891 bbox
```

Rekomendasi final:

```text
reports/percentile_paper_critical_anomalies.csv
```

Alasan:

1. Paling konservatif.
2. Mengikuti metode percentile threshold dari paper.
3. Hanya top 1% paling ekstrem per konteks.
4. Lebih efektif mengurangi false positive daripada threshold global bbox.

---

## 14. Xue-Light Calibration Diagnostic

Xue-light mengecek apakah dataset punya dukungan garis lurus panjang untuk pendekatan plumb-line/fisheye calibration.

Script:

```text
scripts/xue_light_calibration.py
```

Metode jika OpenCV tersedia:

```text
gray image
GaussianBlur
Canny edge
HoughLinesP
count long lines >= 250 px
```

Hasil terakhir:

```text
status = completed
sampled_images = 20
total_line_segments = 27,852
total_long_line_segments = 22,074
long_line_support_per_image = 1,103.7
weighted_mean_abs_angle_deg = 32.44
recommendation = Feasible for Xue/plumb-line calibration
```

File:

```text
configs/xue_light_calibration.json
reports/xue_light_calibration.json
```

Interpretasi:

1. Dataset punya dukungan garis panjang yang kuat.
2. Pendekatan Xue/plumb-line feasible untuk pengembangan berikutnya.
3. Tahap sekarang belum full fisheye rectification.
4. Belum ada parameter undistortion `k1/k2/k3` atau transformasi bbox dari Xue-light.

---

## 15. Urutan Pipeline Final

Command:

```powershell
python scripts/run_pipeline.py
```

Urutan script:

```text
scripts/audit_dataset.py
scripts/extract_bbox_features.py
scripts/estimate_weight_anomalies.py
scripts/compare_camera_corrections.py
scripts/xue_light_calibration.py
scripts/image_level_anomaly.py
```

Alur:

```text
YOLO labels
  -> bbox features
  -> Cobb500 age/week anchor
  -> original weight estimate
  -> radial correction
  -> depth-light correction
  -> radial_depth_median estimate
  -> image-level context
  -> robust/MAD anomaly
  -> percentile P97/P99 anomaly
```

---

## 16. Referensi yang Digunakan di Final Best Calibration

### 16.1 Cobb500 Broiler Performance & Nutrition Supplement 2022

File:

```text
2022-Cobb500-Broiler-Performance-Nutrition-Supplement_copy.pdf
```

Peran:

1. Standar target berat broiler.
2. Anchor estimasi gram per minggu.
3. Dasar pembandingan image/flock terhadap standar performa.

Dipakai untuk target Cobb500 As-Hatched:

```text
W1 = 202 g
W2 = 570 g
W3 = 1116 g
W4 = 1783 g
W5 = 2521 g
W6 = 3278 g
```

Status: dipakai sebagai standar utama final.

### 16.2 Automated Precision Weighing: Leveraging 2D Video Feature Analysis and Machine Learning for Live Body Weight Estimation of Broiler Chickens

File:

```text
1-s2.0-S2772375525000279-main_copy.pdf
```

Peran:

1. Mendukung penggunaan fitur 2D untuk estimasi berat broiler.
2. Mendukung `minor_axis` sebagai fitur utama.
3. Mendukung umur/age sebagai fitur penting.
4. Menjadi dasar formula yang memprioritaskan `minor_axis`.

Implementasi:

```text
estimated_weight_g = 0.7 * est_minor + 0.3 * est_area
```

Status: dipakai untuk desain fitur visual dan estimasi relatif.

### 16.3 DaFIR: Distortion-Aware Representation Learning for Fisheye Image Rectification

File:

```text
DaFIR_Distortion-Aware_Representation_Learning_for_Fisheye_Image_Rectification.pdf
```

Peran:

1. Dasar bahwa distorsi fisheye/radial berkaitan dengan jarak pixel dari pusat gambar.
2. Mendukung fitur `radius_norm`.
3. Mendukung radial correction berbasis bin radius.

Implementasi ringan:

```text
radius_norm = distance(bbox_center, image_center) / max_image_radius
radial_corrected_minor_axis = minor_axis / radial_scale_factor
```

Status: dipakai sebagai inspirasi koreksi radial ringan, bukan full DaFIR rectification.

### 16.4 An End-to-End Depth-Based Pipeline for Selfie Image Rectification

File:

```text
An_End-to-End_Depth-Based_Pipeline_for_Selfie_Image_Rectification.pdf
```

Peran:

1. Dasar bahwa ukuran visual dipengaruhi perspektif/depth.
2. Mendukung normalisasi ukuran visual terhadap posisi/depth.
3. Menginspirasi `bottom_y_norm` sebagai proxy depth-light.

Implementasi ringan:

```text
bottom_y_norm = posisi bawah bbox / tinggi gambar
radial_depth_corrected_minor_axis = radial_corrected_minor_axis / perspective_scale_factor
```

Status: dipakai sebagai inspirasi depth-light correction, bukan full depth reprojection.

### 16.5 Extending Foundational Monocular Depth Estimators to Fisheye Cameras with Calibration Tokens

File:

```text
Gangopadhyay_Extending_Foundational_Monocular_Depth_Estimators_to_Fisheye_Cameras_with_Calibration_ICCV_2025_paper.pdf
```

Peran:

1. Menguatkan bahwa depth estimator biasa bisa bias pada kamera fisheye tanpa kalibrasi.
2. Mendukung kebutuhan correction yang sadar intrinsic/distortion camera.
3. Menjadi dasar batasan bahwa `bottom_y_norm` hanya proxy ringan.

Status: dipakai sebagai referensi validitas dan keterbatasan metode depth-light.

### 16.6 Learning to Calibrate Straight Lines for Fisheye Image Rectification

File:

```text
Xue_Learning_to_Calibrate_Straight_Lines_for_Fisheye_Image_Rectification_CVPR_2019_paper.pdf
```

Peran:

1. Dasar diagnostic Xue-light/plumb-line calibration.
2. Menggunakan asumsi garis lurus dunia nyata harus menjadi lurus setelah rectification.
3. Cocok untuk image kandang dengan feeder, pipa, dinding, atau garis lantai.

Implementasi:

```text
Canny edge detection
HoughLinesP
long line segment counting
```

Status: dipakai untuk diagnostic feasibility, belum full undistortion.

### 16.7 Comparing Threshold Selection Methods for Network Anomaly Detection

File:

```text
Comparing_Threshold_Selection_Methods_for_Network_Anomaly_Detection.pdf
```

Peran:

1. Dasar metode threshold percentile.
2. Paper mendefinisikan `T = perc(k, X)`.
3. Project mengadaptasi `k = 97` untuk warning dan `k = 99` untuk critical.
4. Menjadi metode final individual anomaly yang paling konservatif.

Implementasi final:

```text
X = abs(log(radial_depth_median_estimated_weight_g / image_median_weight_g))
warning  = score >= P97
critical = score >= P99
```

Status: dipakai untuk threshold anomaly final.

---

## 17. Referensi yang Dipertimbangkan tetapi Tidak Dipakai sebagai Standar Final

### Ross 308 Performance Objectives 2022

File:

```text
RossxRoss308-BroilerPerformanceObjectives2022-EN_copy.pdf
```

Status:

```text
Tidak dipakai sebagai standar final.
```

Alasan:

1. Metadata dataset tidak memastikan strain ayam.
2. Pipeline utama memakai Cobb500.
3. Menggabungkan Cobb500 dan Ross 308 membuat interpretasi hasil bercampur.
4. Keputusan final project memakai Cobb500 saja.

---

## 18. Keterbatasan Validitas

1. Belum ada data timbang manual per ayam.

   Estimasi ini relatif terhadap Cobb500 dan fitur visual, belum regresi berat aktual terkalibrasi.

2. Tidak ada tracking ayam antar frame.

   Setiap bbox dianggap observasi individu. Jika video berisi frame berurutan, satu ayam bisa terhitung lebih dari sekali.

3. Koreksi radial dan depth masih pendekatan ringan.

   Belum ada parameter kamera fisik, homography lantai, depth map nyata, atau undistortion penuh.

4. Xue-light masih diagnostic.

   `Feasible for Xue/plumb-line calibration` berarti dataset mendukung pengembangan lanjut, bukan berarti rectification penuh sudah diterapkan.

5. Percentile P99 adalah anomaly relatif.

   P99 berarti top 1% paling ekstrem dalam konteks image/house-week, bukan otomatis sakit, cacat, atau berat aktual salah.

6. Cobb500 adalah asumsi standar final.

   Jika strain aktual bukan Cobb500, interpretasi terhadap standar performa perlu disesuaikan.

---

## 19. Rekomendasi Penggunaan

### Laporan utama individual anomaly

Gunakan:

```text
reports/percentile_paper_critical_anomalies.csv
```

Alasan: paling konservatif, mengikuti metode paper, dan hanya berisi P99/top 1% paling ekstrem.

### Review kandidat lebih luas

Gunakan:

```text
reports/percentile_paper_individual_anomalies.csv
```

Alasan: berisi P97+, top 3% observasi paling ekstrem untuk inspeksi manual lebih luas.

### Monitoring flock/image

Gunakan:

```text
reports/image_level_anomalies.csv
```

Alasan: membandingkan rata-rata image ke Cobb500 dan memberi indikator uniformity/CV.

### Analisis bias kamera

Gunakan:

```text
reports/anomaly_baseline_comparison.html
reports/correction_factors.json
```

Alasan: menunjukkan efek original vs radial vs radial-depth dan membantu membaca apakah anomaly dipengaruhi posisi kamera.

---

## 20. Langkah Lanjutan yang Disarankan

Prioritas tinggi:

1. Tambahkan data berat aktual manual untuk kalibrasi regresi.
2. Tambahkan tracking ayam antar frame agar satu ayam tidak dihitung berulang.
3. Lanjutkan Xue-light dari diagnostic ke estimasi parameter radial distortion.
4. Jika parameter radial stabil, transformasi bbox memakai correction map/undistortion.
5. Tambahkan homography lantai untuk koreksi perspektif lebih baik.

Prioritas menengah:

1. Tambahkan model regresi:

```text
weight ~ minor_axis + age + radius_norm + bottom_y_norm
```

2. Bandingkan Linear, Robust Linear, Random Forest, dan Gradient Boosting jika data timbang tersedia.
3. Tambahkan aggregation per image/video/session.
4. Tambahkan dashboard visual.

---

## 21. Kesimpulan

Final best calibration project ini adalah kombinasi:

```text
Cobb500 As-Hatched
+ minor_axis dan ellipse_area sebagai fitur visual
+ baseline house-week
+ radial correction dari radius_norm
+ perspective-light correction dari bottom_y_norm
+ median baseline robust
+ image-context median/MAD
+ percentile threshold P97/P99
```

Output final paling direkomendasikan:

```text
reports/percentile_paper_critical_anomalies.csv
```

Nilai final:

```text
P99 critical anomaly = 3,667 bbox = 1.14%
```
