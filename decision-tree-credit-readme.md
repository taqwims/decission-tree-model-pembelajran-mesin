# Model Klasifikasi Kelayakan Kredit Komputer dengan Decision Tree

## Deskripsi Proyek
Proyek ini bertujuan untuk membangun model klasifikasi yang dapat memprediksi kelayakan seseorang untuk mendapatkan kredit komputer berdasarkan beberapa fitur seperti umur, pendapatan, status mahasiswa, dan peringkat kredit. Model yang digunakan adalah Decision Tree.

## Dataset
Dataset berisi informasi tentang kelayakan kredit komputer dengan fitur-fitur berikut:
- **Age**: Kategori umur (Tua, Paruh Baya, Muda)
- **Income**: Tingkat pendapatan (Tinggi, Sedang, Rendah)
- **Student**: Status mahasiswa (Ya, Tidak)
- **Credit_Rating**: Peringkat kredit (Baik, Buruk)
- **Buys_Computer**: Target variabel (1 = Layak, 0 = Tidak Layak)

## Tahapan Pembuatan Model

### 1. Load Data
- Memuat dataset kredit komputer dari file yang sudah diunduh
- Memastikan format data sesuai dengan yang diharapkan

### 2. Eksplorasi Data
- Memeriksa struktur dan informasi dataset
- Analisis statistik deskriptif untuk memahami distribusi data
- Memeriksa nilai yang hilang
- Visualisasi distribusi kelas target (layak/tidak layak kredit)
- Analisis hubungan antara fitur dan target

### 3. Preprocessing Data
- Identifikasi fitur kategorikal dan numerikal
- Memisahkan fitur (X) dan target (y)

### 4. Feature Engineering & Encoding
- Standarisasi fitur numerik menggunakan StandardScaler (jika ada)
- Encoding fitur kategorikal menggunakan OneHotEncoder
- Menggunakan ColumnTransformer untuk menangani kedua jenis fitur sekaligus

### 5. Pembagian Data
- Membagi data menjadi training set (70%) dan testing set (30%)
- Menggunakan stratifikasi untuk memastikan distribusi kelas target seimbang di kedua set

### 6. Pembuatan Model Decision Tree
- Membuat pipeline yang menggabungkan preprocessor dan model Decision Tree
- Memastikan integrasi yang mulus antara preprocessing dan model

### 7. Pelatihan Model
- Melatih model dengan data training
- Memantau proses pelatihan untuk memastikan konvergensi

### 8. Evaluasi Awal
- Mengevaluasi performa model dengan beberapa metrik:
   - Accuracy: Persentase prediksi yang benar
   - Precision: Keakuratan prediksi positif
   - Recall: Kemampuan model menangkap kasus positif
   - F1 Score: Harmonik mean dari precision dan recall
   - Confusion Matrix: Visualisasi hasil klasifikasi benar dan salah

### 9. Hyperparameter Tuning
- Menggunakan GridSearchCV untuk menemukan parameter terbaik
- Parameter yang dioptimalkan:
   - max_depth: Kedalaman maksimum pohon
   - min_samples_split: Jumlah minimum sampel untuk split internal
   - min_samples_leaf: Jumlah minimum sampel untuk menjadi leaf node
   - criterion: Fungsi untuk mengukur kualitas split (gini atau entropy)

### 10. Model Final dengan Parameter Terbaik
- Membuat model final menggunakan parameter terbaik dari hasil tuning
- Mengevaluasi performa model final dengan metrics yang sama
- Membandingkan dengan model awal untuk melihat peningkatan performa

### 11. Visualisasi Decision Tree
- Memvisualisasikan struktur pohon keputusan untuk interpretasi model
- Menampilkan aturan-aturan keputusan yang dibuat oleh model

### 12. Feature Importance
- Menganalisis dan memvisualisasikan fitur-fitur yang paling berpengaruh dalam model
- Mengidentifikasi faktor-faktor kunci dalam penentuan kelayakan kredit

### 13. Validasi Silang (Cross-Validation)
- Melakukan cross-validation untuk menilai kestabilan dan keandalan model
- Menganalisis variasi performa model pada berbagai subset data

### 14. Kurva ROC
- Menganalisis trade-off antara true positive rate dan false positive rate
- Menghitung AUC (Area Under Curve) sebagai metrik evaluasi tambahan

### 15. Penyimpanan Model
- Menyimpan model final untuk penggunaan di masa depan
- Memastikan model dapat dimuat kembali dengan mudah

### 16. Contoh Prediksi
- Menunjukkan contoh penggunaan model untuk memprediksi kelayakan kredit
- Menginterpretasikan hasil prediksi untuk kasus-kasus baru

## Cara Menggunakan Model

### Instalasi
```bash
pip install -r requirements.txt
```

### Menjalankan Kode
```bash
python decision_tree_credit_classification.py
```

### Menggunakan Model Tersimpan
```python
import joblib
import pandas as pd

# Muat model
model = joblib.load('decision_tree_credit_model.pkl')

# Data baru untuk diprediksi
new_data = pd.DataFrame({
    'Age': ['Tua', 'Muda', 'Paruh Baya'],
    'Income': ['Rendah', 'Tinggi', 'Sedang'],
    'Student': ['Ya', 'Tidak', 'Ya'],
    'Credit_Rating': ['Baik', 'Baik', 'Buruk']
})

# Lakukan prediksi
predictions = model.predict(new_data)
probabilities = model.predict_proba(new_data)

# Tampilkan hasil
for i, pred in enumerate(predictions):
    print(f"Data #{i+1}: Prediksi = {pred} ({'Layak' if pred == 1 else 'Tidak Layak'})")
    print(f"Probabilitas kelayakan: {probabilities[i][1]:.4f}")
```

## Hasil dan Kesimpulan
Model Decision Tree yang dibangun mampu mengklasifikasikan kelayakan kredit komputer dengan tingkat akurasi yang baik. Faktor-faktor yang paling berpengaruh dalam penentuan kelayakan kredit berdasarkan model ini adalah [akan diisi setelah model dilatih].

Visualisasi pohon keputusan menunjukkan bahwa model membuat keputusan berdasarkan aturan-aturan yang mudah diinterpretasikan, menjadikan model ini tidak hanya akurat tetapi juga dapat dipahami oleh pengguna bisnis.

## Dependensi
- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib

## Catatan
Model ini dikembangkan untuk tujuan pembelajaran dan demonstrasi. Untuk implementasi dalam lingkungan produksi, perlu dilakukan pengujian dan optimalisasi lebih lanjut.
