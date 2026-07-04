# Briefing untuk Dosen — Arah Riset Generalizability MASSA AYAM

Tanggal: 2026-06-27

## 1. Ringkasan 1 Menit

Proyek awal memakai dataset PIO untuk estimasi/anomali berat broiler berbasis bounding box YOLO dan standar Cobb500. Karena data timbang aktual atau metadata umur tidak tersedia pada dataset eksternal publik, arah riset diperluas menjadi dua mode:

1. **Mode absolut pada PIO**  
   Menggunakan metadata week dari filename dan standar Cobb500. Ini tetap menjadi baseline utama untuk estimasi berat relatif.

2. **Mode relatif pada dataset eksternal**  
   Tidak mengestimasi berat gram. Menguji apakah fitur visual bbox, koreksi kamera, dan threshold percentile P97/P99 tetap stabil pada dataset lain.

Kesimpulan awal: pipeline berhasil berjalan pada PIO + 5 dataset eksternal. Pada dataset eksternal yang dense, threshold P97/P99 menghasilkan kandidat anomali yang terkendali, sehingga arah riset generalizability layak dilanjutkan sebagai **relative visual-anomaly generalization**, bukan absolute body-weight prediction.

---

## 2. Apa yang Sudah Dikerjakan

### Dataset baseline

- PIO dataset sudah tervalidasi.
- 1,487 gambar.
- 327,283 bbox valid.
- Metadata week tersedia dari filename.
- Pipeline Cobb500 lama tetap berjalan.

### Dataset eksternal

Dataset publik yang berhasil diproses:

| Dataset | Source | Images | BBox | Kegunaan |
|---|---|---:|---:|---|
| NESTLER | Zenodo | 480 | 4,043 | domain berbeda, video→YOLO |
| Broiler Healthy & Sick | Roboflow | 491 | 491 | sparse robustness check |
| Broiler Instance Segmentation | Roboflow | 200 | 10,570 | broiler dense, kuat untuk generalization |
| Chicken Count | Roboflow | 178 | 3,646 | mixed resolution/domain shift |
| FUM Chicken Detection | Roboflow | 326 | 29,355 | dense high-density detection |

Dataset eksternal ini tidak punya metadata umur/berat, sehingga diproses dalam mode relatif.

---

## 3. Hasil Utama

| Dataset | P97+ candidate | P99+ critical | Interpretasi |
|---|---:|---:|---|
| NESTLER | 3.02% | 1.01% | Stabil; cocok untuk uji domain berbeda |
| Broiler Healthy & Sick | 0.00% | 0.00% | Terlalu sparse; tidak cocok untuk image-level flock anomaly |
| Broiler Instance Segmentation | 3.78% | 1.89% | Kuat untuk generalizability |
| Chicken Count | 3.73% | 2.58% | Mixed resolution; domain shift tinggi |
| FUM Chicken Detection | 3.52% | 1.48% | Kuat untuk high-density generalizability |

Interpretasi penting:

- P97+ umumnya sekitar 3–4%.
- P99+ umumnya sekitar 1–2% pada dataset dense.
- Ini sesuai tujuan percentile threshold: mengambil kandidat paling ekstrem, bukan semua objek berbeda ukuran.
- Dataset sparse tidak bisa dipakai untuk klaim flock-level karena 1 bbox/image membuat median image tidak bermakna.

---

## 4. Klaim yang Aman

Boleh diklaim:

1. Pipeline dapat membaca dan memproses beberapa dataset publik selain PIO.
2. Fitur visual bbox dapat diekstrak konsisten lintas dataset.
3. Mode relative anomaly memungkinkan evaluasi generalizability tanpa data timbang aktual.
4. Pada dataset dense, threshold P97/P99 tetap menghasilkan rate kandidat yang terkendali.
5. Pipeline PIO lama tetap valid untuk mode Cobb500 karena metadata week tersedia.

Tidak boleh diklaim:

1. Berat aktual berhasil diprediksi pada dataset eksternal.
2. Ayam yang terdeteksi anomali pasti sakit/kurus secara biologis.
3. Model sudah universal untuk semua kondisi kandang.
4. Dataset sparse setara dengan dataset high-density commercial farm.

---

## 5. Kalimat Siap Pakai untuk Bimbingan

> Karena dataset eksternal publik tidak menyediakan metadata umur ayam maupun ground-truth berat, evaluasi generalisasi dilakukan dalam dua mode. Dataset PIO digunakan untuk mode absolut berbasis Cobb500, sedangkan dataset eksternal digunakan untuk mode relative visual-anomaly. Dengan cara ini, penelitian tetap dapat menguji apakah fitur bbox, koreksi kamera, dan threshold percentile stabil pada berbagai domain visual tanpa membuat klaim berat aktual yang tidak didukung data.

Versi Inggris:

> Since public external poultry datasets generally do not provide bird age or ground-truth body weight, cross-dataset evaluation is conducted in relative visual-anomaly mode rather than absolute weight prediction mode. This allows the pipeline's visual features, camera correction, and percentile-based anomaly thresholds to be tested under different dataset conditions without overclaiming actual body-weight estimation.

---

## 6. File yang Dibuka Saat Bimbingan

### Paling penting

```text
RESEARCH_DIRECTION_GENERALIZABILITY.md
DATASET_GENERALIZABILITY_TABLE.md
PHASE_2_RELATIVE_GENERALIZATION.md
reports/external/cross_dataset_relative_report.html
reports/external/cross_dataset_relative_summary.csv
```

### Jika dosen ingin cek teknis

```text
configs/datasets/external_datasets.json
scripts/run_external_relative_pipeline.py
scripts/relative_anomaly_pipeline.py
scripts/extract_external_bbox_features.py
CHECKPOINT.md
```

### Output baseline PIO

```text
reports/percentile_paper_anomaly_report.html
reports/image_level_anomaly_report.html
reports/anomaly_baseline_comparison.html
```

---

## 7. Rekomendasi Next Experiment

Prioritas berikutnya:

1. **Dense-only analysis**  
   Jalankan analisis hanya pada image dengan `bbox_count >= 20`. Ini mengurangi bias dataset sparse.

2. **Resolution/domain-shift analysis**  
   Bandingkan dataset uniform resolution vs mixed resolution untuk melihat dampak resolusi pada anomaly score.

3. **Cross-dataset figure/table**  
   Buat grafik P97/P99 rate, median bbox/image, dan image CV per dataset untuk dimasukkan ke proposal/laporan.

4. **Jika data timbang aktual didapat**  
   Tambahkan regresi berat aktual:

```text
weight ~ minor_axis + ellipse_area + age + radius_norm + bottom_y_norm
```

---

## 8. Status Akhir

Fase 1.3 selesai:

- Dataset publik ditemukan, diunduh, dikonversi, divalidasi.

Fase 2 selesai:

- Pipeline multi-dataset relative anomaly dibuat.
- Output cross-dataset dibuat.
- PIO pipeline lama diverifikasi tetap berjalan.

Arah riset saat ini siap dipresentasikan sebagai:

```text
Generalizability testing of a broiler visual-anomaly pipeline across public poultry detection datasets.
```
