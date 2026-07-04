# Review Metodologis — Generalisasi Dataset (MASSA AYAM Phase 2)

Tanggal: 2026-06-28
Status: dokumen review/kritik metodologis + saran penyelesaian (belum diimplementasikan ke kode)

Dokumen ini fokus **hanya pada masalah metodologis** evaluasi generalisasi lintas dataset.
Masalah teknis non-metodologis (path data putus, runner menimpa hasil) sudah dibahas terpisah
dan tidak diulang di sini.

---

## 0. Prinsip pemandu

> **Ukur sesuatu yang BISA gagal.**
> Sebuah metrik generalisasi hanya bermakna kalau secara prinsip ia bisa keluar jelek.
> Rate P97/P99 tidak pernah bisa jelek (selalu ±3% / ±1% karena sifat percentile).
> AUROC, transfer threshold yang dibekukan, dan validasi label nyata BISA jelek —
> itulah yang harus dilaporkan sebagai bukti generalisasi.

Akar dari kelima masalah di bawah adalah **dua keputusan desain yang belum ditetapkan**:

1. **Unit analisis** — apa yang dihitung sebagai satu observasi independen (bbox / image / track / video / dataset)?
2. **Rezim kalibrasi & klaim** — pipeline yang *dibekukan dari PIO lalu ditransfer*, atau prosedur yang *fit-ulang per dataset*?

Begitu dua ini ditetapkan, lima masalah berubah dari "kelemahan" menjadi "tugas konkret".

---

## 1. Ringkasan masalah

| Kode | Masalah | Tingkat | Inti |
|---|---|---|---|
| C1 | Tidak ada model/prediksi yang diuji | Fundamental | Yang dievaluasi heuristik di atas label ground-truth, bukan performa model |
| C2 | Rate P97/P99 tautologis | Tinggi | Percentile yang di-fit ke data sendiri selalu menandai ~3%/1% — bukan bukti |
| C3 | `abnormal_image_rate` didominasi jumlah sampel | Tinggi | Angka "100% abnormal" menyesatkan; flag data-tak-cukup dicampur flag anomali |
| C4 | Koreksi & konteks di-fit ulang per dataset | Sedang | Angka lintas-dataset tidak apple-to-apple; konteks percentile bercampur |
| C5 | Korelasi temporal (nestler) | Sedang | 4043 bbox bukan 4043 observasi independen; frame berurutan |

---

## 2. Detail masalah dan solusi

### C1 — Tidak ada model/prediksi: yang diuji heuristik di atas ground-truth

**Masalah.**
Seluruh pipeline eksternal membaca **label YOLO ground-truth** tiap dataset, bukan keluaran sebuah
model. Tidak ada mAP, tidak ada MAE berat vs timbangan, tidak ada akurasi. Jadi yang dievaluasi
adalah **prosedur anomali heuristik**, bukan "performa model lintas dataset".

**Bukti/lokasi.**
- `scripts/extract_external_bbox_features.py` → input = file `.txt` label dataset.
- `scripts/relative_anomaly_pipeline.py` → tidak ada pemanggilan model/inferensi di mana pun.

**Dampak.** Istilah "generalizability/performa model" di judul & tabel mudah di-challenge penguji.

**Solusi.**
1. **Reframe jujur**: berat absolut (Cobb500) **tidak bisa** divalidasi di luar PIO karena tak ada
   ground-truth timbangan. Nyatakan ini sebagai *limitation*, bukan disembunyikan.
2. **Pindahkan klaim ke komponen yang memang bisa diuji**: kemampuan *mendeteksi outlier ukuran*
   (lihat C2 + Bagian 4) — bukan "ketepatan berat".
3. **Buat ground-truth** supaya ada angka performa nyata (injeksi sintetis + label sehat/sakit;
   lihat E2, E3).

---

### C2 — Rate P97/P99 tautologis

**Masalah.**
Threshold P97/P99 dihitung **dari data dataset itu sendiri**, lalu dihitung berapa persen yang
melampauinya. Secara definisi top-3% = 3% dan top-1% = 1% pada konteks cukup besar — berapa pun
datasetnya. Temuan "P97≈3%, P99≈1% di semua dataset" adalah sifat percentile, bukan bukti generalisasi.

**Bukti/lokasi.**
- `scripts/relative_anomaly_pipeline.py` baris ~209–221 (`thresholds[...] = percentile(scores, 0.97/0.99)`
  lalu `percentile_level()` membandingkan skor ke threshold dari grup yang sama).

**Dampak.** Metrik headline lintas-dataset kosong makna; tabel terlihat "stabil" secara artifisial.

**Solusi.** Ganti rate percentile dengan metrik **bebas-threshold** dan **transfer**:
- **Transfer threshold (E1):** ambil nilai skor P99 dari PIO, terapkan ke dataset lain **tanpa
  fit ulang**. Fraksi yang ke-flag jadi besaran yang *bisa* salah (0% atau 40% = sinyal domain shift).
- **AUROC (E2):** bebas threshold sepenuhnya → tautologi hilang.
- Kalau tetap ingin melaporkan threshold, laporkan **nilai skor di P99** (besarannya), bukan rate-nya,
  dan cek apakah nilai itu konsisten antar dataset dense.

---

### C3 — `abnormal_image_rate` didominasi jumlah sampel

**Masalah.**
Flag `low_sample_count` (bbox/image < 20) digabung dengan flag anomali ke satu boolean
`image_is_abnormal`. Karena nestler median 9 bbox/image dan broiler_healthy_sick 1 bbox/image,
hampir semua image otomatis "abnormal" hanya karena sedikit bbox — bukan karena flock anomali.
Hasilnya `abnormal_image_rate = 100%` (nestler, broiler_healthy_sick) dan 89–98% lainnya.

**Bukti/lokasi.**
- `scripts/relative_anomaly_pipeline.py`:
  - `image_quality_flags()` baris ~114–126 (mencampur `low_sample_count` dan `cv_gt_20/30`).
  - baris ~177 `image_is_abnormal = flags != "normal_image"`.
  - ringkasan `abnormal_images` / `abnormal_image_rate_pct` baris ~234, ~256.

**Dampak.** "100% abnormal" di tabel skripsi sangat mudah disalahartikan penguji.

**Solusi.**
1. **Pisahkan dua jenis flag**: kecukupan data (count) vs anomali sesungguhnya (CV). Jangan
   dikolapskan ke satu boolean.
2. **Kriteria inklusi**: hitung rate anomali **hanya pada subset image yang memenuhi syarat**
   (mis. ≥20 bbox), dan laporkan *coverage*-nya ("X% image memenuhi syarat"). Image kurang-padat
   **dikeluarkan**, bukan dihitung abnormal.
3. **Ambang CV jangan arbitrer**: ambang CV>20% perlu **dijustifikasi dari PIO** (berapa CV flock
   sehat di PIO?) atau ganti dengan melaporkan **distribusi CV**, bukan biner.

---

### C4 — Koreksi & konteks di-fit ulang per dataset → tidak apple-to-apple

**Masalah.**
Faktor radial/depth dihitung ulang dari tiap dataset, dan **tipe konteks percentile berganti-ganti
di dalam satu dataset** (per-image kalau ≥20 bbox, kalau tidak per-dataset-split). Untuk chicken_count
konteksnya campur (1153 split + 2493 image). Akibatnya threshold bermakna berbeda antar baris, dan
angka antar dataset tidak bisa dibandingkan langsung. Catatan: untuk chicken_count
`radial_correction_effect = -0.98` (koreksi malah **memperburuk** CV) — diperlakukan seolah koreksi
selalu membantu.

**Bukti/lokasi.**
- `scripts/relative_anomaly_pipeline.py`:
  - `add_camera_corrections()` baris ~58–111 (fit ulang tiap pemanggilan).
  - context-switch per-baris baris ~190 (`context_key = image_key if count>=20 else fallback`).

**Dampak.** Perbandingan lintas-dataset rapuh; klaim "stabil" tidak terdefinisi dengan baik.

**Solusi.**
1. **Pilah "frozen" per-lapis** (penting — lihat Bagian 4.2):
   - **Bekukan threshold operasi dari PIO**, **tapi koreksi kamera tetap fit-ulang per dataset**
     (koreksi memang spesifik-kamera; membekukannya ke kamera lain doomed & tak informatif).
2. **Hentikan context-switching diam-diam**: satu kebijakan konteks per eksperimen. Kalau perlu
   menangani sparse, laporkan **dua kolom terpisah** (image-context vs split-context), jangan dicampur.
3. **Pakai besaran skala-invarian** untuk semua perbandingan lintas-dataset (rasio, CV, area
   ternormalisasi) — jangan piksel mentah, karena resolusi berbeda jauh.

---

### C5 — Korelasi temporal (nestler): N efektif jauh < 4043

**Masalah.**
Sampling mengambil **80 frame pertama yang berurutan** per video (`f000000, f000001, ...`). Frame
berurutan = ayam nyaris sama, posisi/ukuran nyaris identik → korelasi temporal tinggi. Tanpa tracking,
ukuran sampel efektif jauh di bawah 4043 bbox.

**Bukti/lokasi.**
- `scripts/prepare_nestler_dataset.py` baris ~318 `frames = sorted(by_frame.keys())[:max_frames]`.
- `track_id` tersedia di `tracks_bbox` (row = `[x1,y1,x2,y2,track_id,assembly_id]`, baris ~114–122)
  tetapi dibuang.

**Dampak.** Klaim "4043 observasi" overstated; satu video ramai bisa mendominasi statistik.

**Solusi.**
1. **Stride sampling**: ganti `[:max_frames]` jadi `frames[::step]` (sebar sepanjang video).
2. **Manfaatkan `track_id`**: ambil **satu / median ukuran per track** → observasi mendekati independen.
3. **Laporkan effective sample size** (jumlah track/video), bukan jumlah bbox mentah; untuk
   perbandingan lintas-dataset, bobot **per-video/clip** agar satu video tidak dominan.

---

## 3. Empat keputusan desain (lapisan solusi yang lebih dalam)

### 4.1 Apa yang sebenarnya di-"generalize"?

Pipeline berlapis; tiap lapis = klaim berbeda. Pilih pembawa klaim secara sadar.

| Lapis | Isi | Klaim kalau ini yang dibawa |
|---|---|---|
| L1 | ekstraksi fitur ukuran | "fitur stabil" — lemah, hampir pasti benar |
| L2 | koreksi kamera (radial+depth) | "koreksi mengurangi varians" — **bisa gagal** (chicken_count −0.98) |
| L3a | estimasi berat Cobb500 | **tidak bisa** diuji eksternal (tak ada timbangan) |
| L3b | anomali ukuran-relatif + percentile | **klaim terkuat & jujur** |

**Rekomendasi:** L3b sebagai pembawa klaim utama; L2 sub-temuan (termasuk kasus gagalnya);
**L3a dinyatakan eksplisit hanya valid di PIO**.

### 4.2 Koreksi kamera itu spesifik-kamera → ubah arti "frozen"

Faktor radial/depth PIO di-fit ke kamera fisheye top-down PIO. Dataset lain (mis. foto HP) punya
geometri kamera berbeda total. Maka:
- **Bekukan threshold operasi saja** (nilai skor P99 PIO) → menguji "apakah sebaran alami
  ukuran-relatif antar flock mirip?" (pertanyaan yang benar-benar terbuka).
- **Koreksi kamera tetap fit-ulang per dataset** (perilaku deployment yang benar).
- Membekukan koreksi = menguji "apakah kameranya identik?" → sudah pasti tidak, jadi buang.

**Rekomendasi hybrid:** koreksi adaptif per-dataset + threshold dibekukan dari PIO.

### 4.3 Strategi ground-truth: seberapa "sintetis" boleh dipercaya?

Kekhawatiran sirkular itu valid. Bedahnya:
- Skor = `abs(log(ukuran/median))`. Skalakan kotak ×2 → skor melonjak → **AUROC absolut pasti
  tinggi & nyaris tak bermakna.**
- **Tapi *degradasi* AUROC antar dataset bermakna**: di chicken_count (CV alami ~78%) anomali
  injeksi tenggelam → AUROC turun. **Sinyalnya = ΔAUROC / ranking antar domain**, bukan angka absolut.

Urutan murah → kredibel:
1. **Magnitudo realistis** (×1.4, bukan ×2) + **evaluasi di threshold-PIO yang dibekukan** →
   laporkan *detection rate* DAN *false-positive rate*. FP-rate **bisa meledak** di dataset
   ber-CV tinggi → falsifiable.
2. **Label nyata healthy/sick** (broiler_healthy_sick) — satu-satunya validitas eksternal asli.
   Sinyal mungkin lemah ("sakit ≠ kecil") — itu pun temuan jujur.
3. **Mini test-set manual** (~150 kotak: runt jelas / oversized jelas / normal) — paling kredibel
   di sidang, mematahkan tuduhan sirkular. Opsional sesuai waktu.

### 4.4 N=5 dataset → studi kasus, bukan uji statistik

- Bingkai sebagai **exploratory/case-study generalization**; laporkan tren & ranking, **bukan**
  p-value lintas-dataset (power tak cukup; dataset bukan sampel acak).
- **Kelompokkan dataset sesuai peran** (lihat Bagian 5).
- Unit independensi: di video, observasi independen = **per-track**, laporkan *effective N*.

---

## 4. Set eksperimen yang benar-benar membuktikan generalisasi

Empat eksperimen ini menggantikan bukti yang sekarang lemah:

- **E1 — Frozen-PIO transfer.** Threshold operasi dari PIO → terapkan ke semua dataset tanpa refit
  (koreksi kamera tetap adaptif). Laporkan fraksi ter-flag + pergeserannya. *(memperbaiki C2, C4)*
- **E2 — Injeksi anomali sintetis → AUROC.** Acak ~5% bbox, skalakan ukuran (×0.4–0.6 atau ×1.4–2.2)
  sebelum ekstraksi; jalankan pipeline penuh; label injeksi=1. Hitung **AUROC** + detection/FP rate
  @threshold-PIO. **Bandingkan AUROC antar dataset.** Caveat: menguji deteksi outlier *geometris*,
  bukan "sakit" klinis; yang dibaca adalah degradasinya. *(memperbaiki C1, C2)*
- **E3 — Validasi label nyata (broiler_healthy_sick).** Pakai label healthy/sick (kelas 0/1) yang
  sekarang di-pool & dibuang sebagai ground-truth anomali: AUROC(score, sick). *(validitas eksternal)*
- **E4 — Deskriptor domain shift** (ganti abnormal-rate): distribusi ukuran (skew/CV), resolusi,
  kepadatan, efek koreksi (termasuk yang negatif). *(memperbaiki C3)*

Output yang diharapkan: tabel lintas-dataset berisi **AUROC, fraksi transfer, deskriptor shift** —
semuanya *bisa* keluar jelek, jadi bermakna — bukan "P97=3%, abnormal=100%".

---

## 5. Kriteria inklusi & pengelompokan dataset

Jangan perlakukan 6 dataset setara. Tetapkan peran eksplisit:

| Peran | Dataset | Dipakai untuk |
|---|---|---|
| **Target generalisasi sah** (dense broiler in-house) | PIO, broiler_instance_seg, chicken_detection_fum | klaim utama |
| **Stress/robustness** (sparse/mixed) | nestler, chicken_count | pendukung/robustness, bukan klaim utama |
| **Validasi label nyata** | broiler_healthy_sick | E3 saja |

Kriteria inklusi target utama (contoh): broiler komersial, median ≥20 bbox/image, resolusi cukup seragam.

---

## 6. Keputusan yang masih perlu diambil

1. **Klaim L3a (berat):** apakah pembimbing mengharapkan "estimasi berat generalize"? Kalau ya,
   ekspektasi perlu diluruskan (data tidak mendukung).
2. **Frozen scope:** setujui hybrid (koreksi adaptif + threshold beku), atau ada alasan pure-frozen?
3. **Ground-truth:** sintetis + healthy/sick cukup, atau sanggup investasi mini test-set manual (~150 kotak)?

---

## 7. Checklist tindakan (urut prioritas)

- [ ] Tetapkan unit analisis & rezim kalibrasi (Bagian 0, keputusan 1–2).
- [ ] Reframe klaim: L3b utama, L3a hanya-PIO, "robustness/diskriminabilitas" bukan "performa berat" (C1).
- [ ] Pisahkan flag kecukupan-data vs anomali; rate hanya di subset ≥20 bbox + coverage (C3).
- [ ] Stride sampling + dedup per-track untuk nestler; laporkan effective N (C5).
- [ ] Implementasi E1 (frozen-PIO transfer) + E4 (deskriptor shift).
- [ ] Implementasi E2 (injeksi sintetis → AUROC, magnitudo realistis, FP-rate @threshold-PIO).
- [ ] Implementasi E3 (validasi healthy/sick).
- [ ] Tulis ulang tabel hasil lintas-dataset memakai metrik baru; perbarui narasi di
      `PHASE_2_RELATIVE_GENERALIZATION.md` dan `SUPERVISOR_BRIEFING_GENERALIZABILITY.md`.
