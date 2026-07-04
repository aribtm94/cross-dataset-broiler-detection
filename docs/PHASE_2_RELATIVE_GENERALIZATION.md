# Phase 2 — Multi-Dataset Relative Generalization

Tanggal: 2026-06-27

## Tujuan

Fase 2 mengadaptasi proyek MASSA AYAM agar bisa menguji generalizability pada dataset eksternal yang tidak memiliki metadata umur/week dan tidak memiliki ground-truth berat.

Prinsip utama:

- **PIO baseline** tetap memakai pipeline lama: Cobb500 + week metadata + estimasi berat gram.
- **Dataset eksternal** memakai **relative anomaly mode**: ukuran visual terkoreksi dibandingkan terhadap konteks image/dataset, bukan berat aktual.

Dengan demikian, hasil eksternal tidak boleh ditulis sebagai “berat ayam dalam gram”, tetapi sebagai:

```text
visual relative-size anomaly
```

---

## Dataset yang diproses

| Dataset | Images | BBox | Median bbox/image | Catatan |
|---|---:|---:|---:|---|
| `nestler_yolo` | 480 | 4,043 | 9.0 | Video NESTLER dikonversi ke YOLO bbox |
| `broiler_healthy_sick` | 491 | 491 | 1.0 | Sparse, 2 kelas, mostly 1 bbox/image |
| `broiler_instance_seg` | 200 | 10,570 | 53.0 | Broiler dense, seg→bbox |
| `chicken_count` | 178 | 3,646 | 18.0 | Mixed-resolution counting dataset |
| `chicken_detection_fum` | 326 | 29,355 | 88.5 | Dense chicken detection, mixed resolutions |

Summary gabungan:

```text
reports/external/cross_dataset_relative_summary.csv
reports/external/cross_dataset_relative_report.html
```

---

## File baru

### Config

```text
configs/datasets/external_datasets.json
```

Berisi daftar dataset eksternal, path, mode, class policy, dan notes.

### Feature extraction

```text
scripts/extract_external_bbox_features.py
```

Output per dataset:

```text
features/external/<dataset_id>/bbox_features.csv
features/external/<dataset_id>/bbox_feature_skips.csv
features/external/<dataset_id>/bbox_feature_summary.json
```

### Relative anomaly pipeline

```text
scripts/relative_anomaly_pipeline.py
```

Output per dataset:

```text
reports/external/<dataset_id>/relative_image_summary.csv
reports/external/<dataset_id>/relative_individual_anomalies.csv
reports/external/<dataset_id>/relative_critical_anomalies.csv
reports/external/<dataset_id>/relative_enriched_features.csv
reports/external/<dataset_id>/relative_anomaly_summary.json
reports/external/<dataset_id>/relative_anomaly_report.html
```

### Runner

```text
scripts/run_external_relative_pipeline.py
```

Menjalankan semua dataset `mode=relative`, lalu membuat summary cross-dataset.

---

## Metode relative anomaly

Untuk setiap bbox, fitur utama:

```text
minor_axis = min(width_px, height_px)
ellipse_area = pi * minor_axis * major_axis
radius_norm = distance(center, image_center) / max_radius
bottom_y_norm = bbox_bottom_y / image_height
```

Koreksi kamera:

```text
radial_corrected_minor_axis
radial_depth_corrected_minor_axis
```

Skor relatif:

```text
relative_to_image_median = radial_depth_corrected_minor_axis / image_median_corrected_minor_axis
robust_z_image = 0.6745 * (x - image_median) / image_mad
relative_percentile_score = abs(log(relative_to_image_median))
```

Threshold:

```text
warning  = score >= P97
critical = score >= P99
```

Fallback context:

- Jika image punya bbox cukup (`>=20`): percentile context = image
- Jika tidak: percentile context = dataset/split

Bug yang sudah diperbaiki:

- Dataset sparse dengan 1 bbox/image menghasilkan `score=0` dan `P97=P99=0`.
- Sekarang `score <= 0` dianggap normal agar semua bbox tidak salah masuk critical.

---

## Hasil cross-dataset

| Dataset | P97+ rate | P99+ rate | Image CV median | Radial effect | Depth effect | Interpretasi |
|---|---:|---:|---:|---:|---:|---|
| `nestler_yolo` | 3.02% | 1.01% | 50.15 | +1.44 | +4.34 | Threshold stabil; sparse/medium density |
| `broiler_healthy_sick` | 0.00% | 0.00% | 0.00 | +0.37 | +2.02 | Limited-use; 1 bbox/image, tidak cocok untuk image-context anomaly |
| `broiler_instance_seg` | 3.78% | 1.89% | 22.96 | +0.57 | +1.37 | Broiler dense; cocok untuk generalization |
| `chicken_count` | 3.73% | 2.58% | 22.82 | -0.98 | +1.65 | Mixed resolutions; domain shift tinggi |
| `chicken_detection_fum` | 3.52% | 1.48% | 34.82 | +0.64 | +0.43 | Dense; cocok untuk high-density generalization |

Catatan:

- P97/P99 umumnya tetap terkendali dekat target konservatif (sekitar 3%/1%), kecuali dataset mixed/sparse.
- Dataset sparse (`broiler_healthy_sick`) tidak valid untuk klaim flock/image-level; hanya valid sebagai check bahwa pipeline tidak crash pada dataset klasifikasi/deteksi sederhana.
- Dataset dense eksternal (`broiler_instance_seg`, `chicken_detection_fum`) paling kuat untuk bukti generalizability.

---

## Verifikasi

Pipeline eksternal berhasil:

```powershell
python scripts/run_external_relative_pipeline.py
```

Pipeline PIO lama tetap berjalan:

```powershell
python scripts/run_pipeline.py
```

Hasil PIO lama tetap konsisten:

```text
features/bbox_features.csv (321,427 rows)
reports/percentile_paper_critical_anomalies.csv (3,667 rows)
```

---

## Narasi riset yang aman

Kalimat yang aman untuk laporan/dosen:

> Karena dataset eksternal tidak menyediakan umur ayam dan data timbang aktual, evaluasi generalizability dilakukan dalam mode anomaly-relatif. Mode ini tidak mengestimasi berat gram absolut, tetapi menguji kestabilan fitur ukuran visual, koreksi kamera, dan threshold percentile P97/P99 pada berbagai domain dataset.

Kesimpulan awal:

> Pipeline dapat dijalankan pada beberapa dataset publik dengan kondisi berbeda. Pada dataset dense, threshold relatif menghasilkan kandidat anomaly yang terkendali, menunjukkan bahwa komponen visual-anomaly dan percentile threshold berpotensi generalizable. Namun, klaim estimasi berat absolut tetap hanya valid untuk dataset yang memiliki metadata umur/standar bobot atau data timbang aktual.
