# Cobb500-based Broiler Weight Anomaly Pipeline

Tujuan: estimasi berat relatif ayam broiler dari anotasi YOLO dan umur minggu, lalu deteksi anomali terhadap target Cobb500 `As Hatched`.

## Input

```text
data/images/train|val/*.jpg
data/labels/train|val/*.txt
data/FilePrefixCode.xlsx
2022-Cobb500-Broiler-Performance-Nutrition-Supplement_copy.pdf
```

Label YOLO:

```text
class x_center y_center width height
```

Mapping umur:

```text
C-W1-XXXX -> Commercial Week 1
P-W1-XXXX -> Prototype Week 1
```

File yang tidak mengikuti pola `C-Wx-*` atau `P-Wx-*` akan di-skip dari estimasi Cobb500 karena umur minggu tidak diketahui.

Asumsi Cobb500:

```text
W1=day7, W2=day14, W3=day21, W4=day28, W5=day35, W6=day42
```

Target Cobb500 `As Hatched`:

```text
W1 202g, W2 570g, W3 1116g, W4 1783g, W5 2521g, W6 3278g
```

## Metode

Fitur per bbox:

```text
minor_axis = min(width_px, height_px)
major_axis = max(width_px, height_px)
ellipse_area = pi * minor_axis * major_axis
radius_norm = jarak pusat bbox ke pusat gambar / radius maksimum gambar
bottom_y_norm = posisi bawah bbox / tinggi gambar
```

`radius_norm` dipakai sebagai koreksi ringan ala DaFIR untuk bias distorsi radial/fisheye. `bottom_y_norm` dipakai sebagai proxy depth/perspective: ayam lebih bawah gambar biasanya lebih dekat kamera.

Estimasi:

```text
est_minor = cobb_weight_g * (minor_axis / median_minor_axis_per_house_week)
est_area = cobb_weight_g * sqrt(ellipse_area / median_ellipse_area_per_house_week)
estimated_weight_g = 0.7 * est_minor + 0.3 * est_area
```

Anomali:

```text
below_week_average: z <= -2 atau <80% median grup
above_week_average: z >= 2 atau >120% median grup
below_cobb_standard: < -10% dari Cobb500
above_cobb_standard: > +10% dari Cobb500
critical_underweight: < -20% dari Cobb500 atau <70% median grup
critical_overweight: > +20% dari Cobb500 atau >130% median grup
```

## Jalankan

```powershell
python scripts/run_pipeline.py
```

## Output

```text
configs/prefix_mapping.csv
configs/cobb500_as_hatched.csv
reports/dataset_audit.json
features/bbox_features.csv
features/weight_estimates.csv
reports/anomalies_individual.csv
reports/anomalies_by_week.csv
reports/anomaly_summary.json
reports/anomaly_report.html
reports/plots/cobb_vs_estimated.svg
reports/overlays/*.svg
features/weight_estimates_compare.csv
reports/anomaly_baseline_comparison.csv
reports/anomaly_baseline_comparison.html
reports/anomaly_baseline_comparison.json
reports/anomalies_consensus.csv
reports/correction_factors.json
configs/xue_light_calibration.json
reports/xue_light_calibration.json
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

## Perbandingan mean/median dan koreksi kamera

`scripts/compare_camera_corrections.py` membuat 6 model pembanding:

```text
original_median
original_mean
radial_median
radial_mean
radial_depth_median
radial_depth_mean
```

`radial_*` memakai koreksi posisi radial sebagai pendekatan ringan dari ide DaFIR bahwa distorsi fisheye bergantung pada radius dari pusat gambar. `radial_depth_*` menambahkan koreksi perspektif ringan dari ide depth-based rectification: ukuran visual dinormalisasi memakai proxy jarak kamera dari `bottom_y_norm`.

Jika `original anomaly` berubah menjadi `corrected normal`, kemungkinan anomali awal disebabkan bias kamera. Jika semua model tetap anomali, kandidat anomali lebih kuat.

## Image-level anomaly final

`scripts/image_level_anomaly.py` memakai model `radial_depth_median` sebagai estimasi visual default, lalu menghitung anomaly pada level image dan individu.

Image-level:

```text
image_mean_weight_g
image_median_weight_g
image_cv_pct
image_cobb_diff_pct
```

Individual-level:

```text
relative_to_image_median = estimated_weight / image_median_weight
robust_z_image = 0.6745 * (estimated_weight - image_median_weight) / MAD_image
```

Threshold final:

```text
image warning:  |diff Cobb| > 10% atau CV > 20%
image critical: |diff Cobb| > 20% atau CV > 30%
individual warning:  relative_to_image_median < 0.75 atau > 1.25, atau |robust_z| > 3.5
individual critical: relative_to_image_median < 0.65 atau > 1.35, atau |robust_z| > 4.5
```

Ini mengurangi false anomaly dari bbox individual karena ayam dalam image yang sama punya bias kamera yang mirip.

## Percentile threshold sesuai paper anomaly threshold selection

Pipeline juga menyediakan threshold persentil mengikuti metode pada paper `Comparing Threshold Selection Methods for Network Anomaly Detection`:

```text
T = perc(k, X)
```

Konfigurasi:

```text
k = 97 untuk warning
k = 99 untuk critical
X = abs(log(radial_depth_median_estimated_weight_g / image_median_weight_g))
```

Karena berat ayam bisa abnormal dua arah (terlalu kecil atau terlalu besar), rasio berat diubah menjadi anomaly score satu arah memakai `abs(log(ratio))`. Makin besar score, makin jauh dari median image.

Konteks threshold:

```text
Jika jumlah bbox dalam image >= 100: pakai P97/P99 per image
Jika jumlah bbox dalam image < 100: fallback ke P97/P99 per house-week
```

Output:

```text
reports/percentile_paper_individual_anomalies.csv
reports/percentile_paper_critical_anomalies.csv
reports/percentile_paper_anomaly_report.html
```

Interpretasi:

```text
P97+ = kandidat anomaly top 3% dalam konteksnya
P99+ = critical anomaly top 1% dalam konteksnya
```

## Xue-light calibration diagnostic

`scripts/xue_light_calibration.py` mengecek apakah gambar kandang punya dukungan garis lurus panjang untuk pendekatan plumb-line/Xue CVPR 2019.

Output:

```text
reports/xue_light_calibration.json
configs/xue_light_calibration.json
```

Jika `opencv-python` tidak tersedia, script menulis status `opencv_unavailable` dan pipeline tetap lanjut. Setelah OpenCV terpasang, hasil terakhir menunjukkan `status=completed`, `sampled_images=20`, `total_long_line_segments=22074`, `long_line_support_per_image=1103.7`, dan rekomendasi `Feasible for Xue/plumb-line calibration`. Tahap ini masih diagnostic, belum full rectification.

## Catatan validitas

Ini estimasi relatif berbasis standar Cobb500, bukan model regresi terkalibrasi dengan data timbang manual. Untuk MAE/MRE seperti paper, tambahkan CSV berat aktual per ayam atau per track.