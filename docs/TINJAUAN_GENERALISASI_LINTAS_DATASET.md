# Tinjauan Pustaka: Generalisasi Metode Deteksi Objek Lintas-Dataset

> Catatan dosen #2 — "carikan paper generalisasi: apa saja yang dibutuhkan untuk men-generalisasi metode agar digunakan di berbagai dataset."
> Disusun dari riset multi-sumber terverifikasi (deep-research, 2026-07-10): 6 angle, 26 sumber di-fetch, 120 klaim diekstrak, **25 klaim diverifikasi adversarial → 24 confirmed, 1 refuted**. Istilah teknis Inggris dipertahankan.

## Ringkasan eksekutif

Untuk membuat detektor ayam broiler (YOLOv8) yang **general lintas dataset** (PIO in-house + 2 dataset Roboflow eksternal), literatur menempatkan masalah ini sebagai **Domain Generalization (DG)** — melatih hanya pada domain sumber agar tahan terhadap domain target yang belum pernah dilihat (*unseen*). Ini berbeda tegas dari **Domain Adaptation (DA/UDA)** yang menuntut akses data target + retraining.

Tiga pilar jawaban:
1. **Augmentasi data yang dipilih cermat + domain randomization** adalah teknik DG-murni paling konsisten terbukti — bahkan sebagai baseline tunggal bisa mengungguli metode SDG kompleks. Syarat kunci: distribusi sintetik harus mendekati domain target.
2. **Untuk distorsi lensa fisheye, konsensus WoodScape adalah "adapt the detector, don't rectify"** — rektifikasi/undistortion justru *menurunkan* mAP (bukti kuantitatif 45.4 tanpa undistortion vs 39.8 rectilinear / 43.7 cylindrical). Ini **mendukung langsung** temuan repo ini bahwa MOWA menurunkan metrik.
3. **Evaluasi** harus memakai protokol cross-dataset / leave-one-dataset-out, dilaporkan **dua arah** (transfer sering asimetris).

---

## 1. Konsep inti: Domain Generalization vs Domain Adaptation

**Masalah fundamental (domain shift).** Detektor berperforma baik *in-distribution* namun turun tajam pada dataset/benchmark berbeda. Pada deteksi, domain shift lebih kompleks daripada klasifikasi karena merambat lintas tahap pipeline (image-level `P_S(I)≠P_T(I)` dan instance-level), bukan satu classifier.
- Deshmukh dkk., 2026, *Generalization Under Scrutiny: Cross-Domain Detection* — **arXiv:2604.08230** (survei)
- Chakraborty dkk., 2026, *Robust Cross-Dataset OD Generalization* — **arXiv:2601.09497**
- Chen dkk., CVPR 2018, *Domain Adaptive Faster R-CNN* — **arXiv:1803.03243**

**DG ≠ DA (taksonomi wajib untuk bab tinjauan pustaka).**
- **DG**: latih **hanya domain sumber**, tanpa akses data target maupun retraining. **Single-Source DGOD (SDGOD)** = setting DG paling murni (satu domain sumber → general ke unseen). Inilah setting kasus skripsi ini (dataset Roboflow tak dipakai saat training).
- **DA/UDA**: butuh **mengumpulkan data target (tak-berlabel)** + retraining per domain target.
- Sumber: **arXiv:2510.19487** (NeurIPS 2025, Causal Visual Prompts SDGOD); **arXiv:2405.14497** (DivAlign, CVPR 2024); **arXiv:2504.20498** (SA-DETR, 2025); **arXiv:2103.03097** (survei *Generalizing to Unseen Domains*); **arXiv:2203.14387** (Zhang dkk., 2022, formalisasi DGOD + benchmark leave-one-dataset-out).

**⚠️ Peringatan klasifikasi (mudah salah di tinjauan pustaka):** Beberapa metode "YOLO cross-domain" populer sebenarnya **DA, bukan DG** — tidak berlaku langsung untuk kasus ini:
- **SSDA-YOLO** (Zhou dkk., CVIU 2023 / **arXiv:2211.02213**): Mean Teacher + scene style transfer, **butuh data target tak-berlabel**.
- **YOLO-G** (Wei dkk., PLOS ONE 2023, DOI 10.1371/journal.pone.0291241): gradient reverse layer + domain classifier adversarial pada backbone YOLOv5-L — **UDA murni**.

---

## 2. Teknik konkret meningkatkan generalisasi lintas-dataset

### 2a. Augmentasi data (terkuat untuk DG murni)
Augmentasi yang **dipilih cermat** adalah baseline DG mandiri yang kuat — dapat mengungguli metode SDGOD SOTA.
- **Mekanisme**: mengganggu pola statistik *low-level* (tekstur/gaya) domain-spesifik sambil mempertahankan konsep *semantik high-level*.
- **Batasan fundamental**: augmentasi hanya efektif bila **distribusi sintetik mendekati distribusi domain target**.
- Catatan penting: augmentasi naif dapat merusak anotasi objek → gunakan varian *object-aware* (**OA-DG, arXiv:2312.12133**).
- Sumber: **arXiv:2405.14497** (DivAlign); **arXiv:2504.20498** (SA-DETR); **arXiv:2312.12133** (OA-DG).

### 2b. Domain Randomization (DR)
Latih pada data dengan variabilitas cukup besar sehingga domain nyata tampak "sekadar variasi lain" — **tanpa** butuh data target saat training (beda dari adversarial DA / GAN translation).
- Sumber fondasi: **arXiv:1703.06907** (Tobin dkk., IROS 2017); **arXiv:2509.15045** (2025, DR via well-tuned augmentation).

### 2c. Augmentasi distorsi geometrik/radial — **paling relevan untuk kamera berdistorsi**
Mengubah citra reguler → **fisheye-like** dengan menyampling koefisien distorsi + focal length acak (= DR geometrik) **meningkatkan** generalisasi detektor fisheye (F1 0.4907 vs 0.4791 tanpa augmentasi fisheye). Sebaliknya, augmentasi standar (vertical flip, copy-paste, multi-scale) tidak memberi perbaikan berarti secara individual.
- Sumber: **arXiv:2507.16254** (Kim & Go, 2025, *Edge-case Synthesis for Fisheye OD*).
- ➡️ **Ini persis arm `radial_retrain` yang sudah menang di repo ini (+0.024 mAP).**

### 2d. Normalisasi domain
**IBN-Net** menggabungkan Instance Normalization (IN) + Batch Normalization (BN) → menaikkan performa sumber sekaligus generalisasi ke domain baru **tanpa fine-tuning / data target**, tanpa biaya komputasi ekstra.
- Sumber: **arXiv:1807.09441** (Pan dkk., ECCV 2018).

### 2e. Test-Time Augmentation (TTA)
Menaikkan robustness saat inferensi **tanpa retraining**: resolusi input ↑, multiscale, flip, digabung via Weighted Boxes Fusion (WBF). YOLOv5x COCO: mAP@0.5:0.95 0.504→0.516.
- Sumber: Ultralytics YOLOv5 TTA docs; **arXiv:2401.01018** (TTA small-object + WBF).
- ➡️ **Ini arm `tta` yang jadi pemenang terbaik di repo ini (+0.035 mAP).**

### 2f. Style transfer / feature alignment adversarial (tergolong **DA**)
Scene style transfer + adversarial GRL meningkatkan deteksi cross-domain substansial (Cityscapes→Foggy 39.9→47.8 mAP) **tetapi butuh akses domain target** → bukan DG murni.

---

## 3. Rektifikasi fisheye (MOWA): membantu atau merugikan?

**Konsensus kuat: MERUGIKAN generalisasi detektor — "adapt the detector, don't rectify".**

- **Rekomendasi eksplisit WoodScape**: "we would like to encourage the community to **adapt** computer vision models for fisheye camera **instead of using naive rectification**." Challenge deteksi fisheye WoodScape (CVPR 2022 OmniCV, 120 tim) mendorong model yang bekerja **native** pada citra fisheye tanpa rektifikasi.
  - **arXiv:1905.01489** (Yogamani dkk., ICCV 2019, WoodScape); **arXiv:2206.12912** (OmniCV challenge, CVPR 2022); **arXiv:2012.02124** (Rashed dkk., WACV 2021).

- **Bukti kuantitatif** (ablation bounding-box): **No undistortion = 45.4 mAP**, mengungguli **rectilinear (39.8)** dan **cylindrical (43.7)**. Penyebab: rectilinear kehilangan field-of-view di periferi + artefak resampling.
  - Angka presisi ada di **versi jurnal extended "Let's Go Bananas", Sensors 2025, DOI 10.3390/s25123735, Table 4** (bukan arXiv v1). Juga **arXiv:2404.13443** (FisheyeDetNet).

- **Mekanisme**: distorsi radial kuat **merusak inductive bias translation-invariance CNN** — objek sama tampak beda tergantung posisi → menaikkan sample complexity. CNN modern (YOLO, Faster R-CNN) justru **dapat mendeteksi langsung** pada citra fisheye mentah.

**➡️ Kaitan langsung dengan skripsi ini:** Temuan repo (semua varian MOWA-rectify < baseline; hanya TTA & radial-augment yang menang) **konsisten penuh** dengan literatur. MOWA sendiri belum muncul di literatur ter-review; kesimpulan tentang MOWA adalah **inferensi dari prinsip umum** "undistortion menurunkan mAP" + bukti A/B repo. Rektifikasi hanya berpeluang membantu **bila diikuti fine-tuning pada citra terektifikasi** (kondisi B' / rectify-both) — dan bahkan itu belum melampaui baseline di repo ini.

---

## 4. Metrik & protokol evaluasi generalisasi lintas-dataset

- **Protokol**: cross-dataset evaluation (train di satu, uji di lainnya) + **leave-one-dataset-out**.
- **Kategorisasi setting** (usulan preprint 2026): *setting-agnostic* (adegan beragam, mis. COCO) vs *setting-specific* (lingkungan sempit, mis. kamera kandang tetap — seperti PIO).
- **Temuan kunci**: transfer **dalam** tipe setting sama relatif stabil; transfer **lintas** tipe turun signifikan dan **sering ASIMETRIS** → **laporkan dua arah** (PIO→Roboflow vs Roboflow→PIO bisa beda).
- **Metrik**: tetap mAP@50 dan mAP@50-95 lintas dataset (seperti yang sudah dipakai repo).
- Sumber: **arXiv:2601.09497**; **arXiv:2604.08230**; **arXiv:2203.14387** (benchmark DGOD 4-setting).

---

## 5. Rekomendasi praktis: deteksi ternak padat (dense small-object) + distorsi lensa

1. **Latih pada citra mentah + augmentasi distorsi radial ala WoodScape** (arm `radial_retrain` repo — terbukti +0.024). **Hindari** pipeline undistortion/MOWA sebagai preprocessing, kecuali diikuti fine-tuning penuh pada domain terektifikasi.
2. **Tambahkan TTA saat inferensi** (arm `tta` repo — pemenang +0.035, terutama broiler +0.105) — gratis, tanpa retraining.
3. **Pertimbangkan arsitektur scale-aware** untuk multiskala unggas: **SFN-YOLO** (**arXiv:2509.17086**, YOLOv8 free-range poultry cross-farm), **YOLO-SDD** (single-class densely populated). Untuk kepadatan ekstrem, **density-map regression** dapat mengungguli bbox detection saat oklusi berat.
4. **Domain randomization koefisien distorsi**: tuning rentang koefisien agar distribusi sintetik mendekati kamera PIO (jangan over-augment).

---

## Caveat (kejujuran ilmiah untuk skripsi)

1. Beberapa sumber (arXiv:2604.08230, arXiv:2601.09497) adalah **preprint 2026, kemungkinan belum peer-reviewed** — kuat sebagai premis/survei, tetapi taksonomi baru (setting-agnostic/specific) kutip sebagai *usulan penulis*, bukan konsensus mapan.
2. Angka rektifikasi (45.4/39.8/43.7) ada di **versi Sensors 2025**, bukan arXiv v1 — kutip versi jurnal.
3. Transfer ke kasus ternak padat bersifat **analogis, bukan identik**: bukti fisheye dari domain automotive/objek tunggal; margin & mekanisme (>180° FoV loss) tak dijamin identik pada kamera kandang PIO.
4. Efek augmentasi fisheye arXiv:2507.16254 kecil (+0.0116 F1, single run tanpa uji signifikansi) → laporkan sebagai peningkatan **moderat**.
5. **MOWA tidak muncul di literatur terverifikasi** — kesimpulan adalah inferensi dari prinsip umum + A/B repo.
6. Klaim "augmentasi → spurious correlations" **DIREFUTASI** (vote 1-2) → **jangan dikutip**.

## Pertanyaan terbuka (untuk diskusi/pengembangan)

1. Apakah 'rectify-both' (fine-tune pada citra MOWA) mengungguli citra mentah + augmentasi radial? (Literatur dukung "adapt", tapi tak uji MOWA langsung pada ternak.)
2. Untuk dense small-object, apakah representasi distortion-aware / deformable-spherical conv (DarSwin **arXiv:2305.00079**, SphereNet) beri gain > augmentasi fisheye pada YOLOv8, dengan overhead berapa?
3. Rentang koefisien distorsi radial optimal untuk DR agar mendekati kamera PIO tanpa over-augment?
4. Bagaimana leave-one-dataset-out menormalkan perbedaan definisi kelas/anotasi antar PIO & Roboflow (label harmonization) agar Δ mAP mencerminkan domain shift, bukan artefak anotasi?

---

## Daftar sitasi ringkas (siap rujuk)

| Topik | Sitasi | ID |
|---|---|---|
| Survei cross-domain detection | Deshmukh dkk. 2026 | arXiv:2604.08230 |
| Cross-dataset OD, setting specificity | Chakraborty dkk. 2026 | arXiv:2601.09497 |
| Domain Adaptive Faster R-CNN | Chen dkk. CVPR 2018 | arXiv:1803.03243 |
| SDGOD causal prompts | NeurIPS 2025 | arXiv:2510.19487 |
| DivAlign (augmentasi DG) | Danish dkk. CVPR 2024 | arXiv:2405.14497 |
| SA-DETR | 2025 | arXiv:2504.20498 |
| OA-DG (object-aware) | 2023 | arXiv:2312.12133 |
| Survei DG | 2021 | arXiv:2103.03097 |
| DGOD benchmark | Zhang dkk. 2022 | arXiv:2203.14387 |
| Domain Randomization | Tobin dkk. IROS 2017 | arXiv:1703.06907 |
| DR via augmentation | 2025 | arXiv:2509.15045 |
| **Fisheye augmentation** | Kim & Go 2025 | arXiv:2507.16254 |
| **WoodScape** | Yogamani dkk. ICCV 2019 | arXiv:1905.01489 |
| OmniCV fisheye challenge | CVPR 2022 | arXiv:2206.12912 |
| Generalized OD Fisheye | Rashed dkk. WACV 2021 | arXiv:2012.02124 |
| "Let's Go Bananas" (angka undistortion) | Sensors 2025 | DOI 10.3390/s25123735 |
| FisheyeDetNet | 2024 | arXiv:2404.13443 |
| IBN-Net (normalisasi) | Pan dkk. ECCV 2018 | arXiv:1807.09441 |
| TTA small-object + WBF | 2023 | arXiv:2401.01018 |
| SSDA-YOLO (DA) | Zhou dkk. CVIU 2023 | arXiv:2211.02213 |
| YOLO-G (DA) | Wei dkk. PLOS ONE 2023 | DOI 10.1371/journal.pone.0291241 |
| SFN-YOLO (poultry) | 2025 | arXiv:2509.17086 |
| DarSwin (distortion-aware) | ICCV 2023 | arXiv:2305.00079 |
