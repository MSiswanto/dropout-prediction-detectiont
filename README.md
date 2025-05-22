# Proyek Akhir: Dropout Prediction of Students

## Business Understanding
Jaya Jaya Institut merupakan salah satu institusi pendidikan perguruan yang telah berdiri sejak tahun 2000 dan telah mencetak banyak lulusan dengan reputasi yang sangat baik. 
Institusi ini mempunyai masalah terkait banyaknya siswa yang tidak menyelesaikan pendidikannya alias dropout.
Jumlah dropout yang tinggi ini tentunya menjadi salah satu masalah yang besar untuk sebuah institusi pendidikan. 
Oleh karena itu, Jaya Jaya Institut ingin mendeteksi secepat mungkin terhadap siswa yang kemungkinan akan melakukan dropout sehingga dapat diberi bimbingan khusus. 
Dashboard juga akan dibuat untuk memudahkan dalam memahami data dan memonitor performa siswa.

### Permasalahan Bisnis
- Tingginya angka dropout
- Menganalisa faktor terbesar penyebab dropout
- Membuat prediksi awal kemungkinan siswa melakukan dropout.
- Membuat dashboard (visualisai) untuk monitoring.

### Cakupan Proyek
Cakupan proyek yang akan dikerjakan:
- menganalisa korelasi antar faktor penyebab tingginya dropout berdasarkan; age at enrollment, gender, marital status, parent's occupation dan lainnya dengan machine learning.
- membuat dashboard terkait dengan Google Looker Studio, Metabase atau Tableau. Google Looker Studio dipilih karena waktu yang terbatas dan limited memory laptop.
- membuat analisa prediksi awal kemungkinan siswa dropout, diklasifikasikan untuk bimbingan khusus.
- membuat visualisasi 

### Persiapan
Langkah-langkah yang akan dilakukan:
- Persiapan Data
- Preprocessing (Encoding, Splitting)
- Modeling (Random Forest / Logistic Regression)
- Evaluasi Model
- Visualisasi: 
  . Confusion Matrix
  . Classification Report
  . Feature Importance
Sumber data: 
  https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/data.csv

## Setup environment:
 1. requirements.txt untuk Proyek Prediksi Dropout
Buat file bernama requirements.txt di root direktori proyek kamu, lalu isi dengan:
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib

2. Struktur Folder yang Direkomendasikan
dropout-prediction/
|
├── app.py                   # Streamlit main app
├── model.pkl                # Trained model (binary classification)
├── label_encoder.pkl        # Label encoder (if needed)
├── top_features.csv         # CSV optional: Top 10 features if precomputed
├── cleaned_data_new.csv     # Dataset (optional for local visualization)
├── requirements.txt         # Environment dependencies
└── README.md                # Project documentation

4. Setup Lokal (Opsional)
Kalau kamu ingin menjalankan lokal sebelum deploy ke Streamlit Cloud:
# Buat virtual environment (opsional tapi direkomendasikan)
  - python -m venv env
  - source env/bin/activate        # macOS/Linux
  - env\Scripts\activate           # Windows
# Install dependencies
  pip install -r requirements.txt
# Jalankan Streamlit app
  streamlit run app.py

4. Upload ke GitHub dan Deploy
Push seluruh folder proyek ke GitHub (pastikan app.py ada di root).
- Masuk ke Streamlit Cloud
- Pilih New App → From GitHub
- Pilih repo → deploy.

## Business Dashboard
Bussiness Dashboard dibuat menggunakan Google Looker Studio. Dari analisa dengan Random Forest, terdapat 10 faktor terbesar penyumbang tingginya jumlah siswa yang dropout:
1.  Curricular units 2nd sem (approved).
2.  Curricular units 2nd sem (grade).
3.  Curricular units 1st sem (approved).
4.  Curricular units 1st sem (grade).
5.  Tuition fees up to date.
6.  Age at enrollment.
7.  Curricular units 2nd sem (evaluations).
8.  Admission grade.
9.  Previous qualification (grade)
10. Course

Berikut merupakan link untuk mengakses dashboard tersebut:
https://lookerstudio.google.com/reporting/d93cb43d-d199-41c9-b088-566bf728d656

## Menjalankan Sistem Machine Learning
Jelaskan cara menjalankan protoype sistem machine learning yang telah dibuat. 
- Geser tombol tiap fitur-fitur di sisi kiri untuk memberi nilai inputan pada algoritma machine learning.
- Klik tombol "Prediksi Dropout" untuk melihat hasilnya, apakah mahasiswa tersebut ada kemungkinan akan dropout atau not dropout.

Link untuk untuk mengakses prototype tersebut:
https://dropout-prediction-detection.streamlit.app/

## Conclusions
Jelaskan konklusi dari proyek yang dikerjakan.
-  Secara umum Curricular units of 2nd & 1st semesters merupakan penyebab utama tingginya dropout, diikuti oleh Admission grade, Age at enrollment, Previous qualification (grade), Tuition fees up to date, debtor dan Scholarship Holder.
-  Admission grade dan previous qualification juga berkorelasi positif dengan tingginya dropout. Perlu diperhatikan lebih student dengan admission-previous qualification grades yang rendah.
- Umur saat sekolah (Age at enrollment) perlu dipertimbangkan juga karena termasuk faktor terbesar penyumbang tingginya dropout.


### Rekomendasi Action Items
Berikan beberapa rekomendasi action items yang harus dilakukan perusahaan guna menyelesaikan permasalahan atau mencapai target mereka.
-   Kurangi atau batasi menerima mahasiswa baru dengan Admission grade dan previous qualification grade yang rendah, karena keduanya berkorelasi positif dengan tingginya dropout. 
-   Perlu adanya pembatasan umur maksimal, karena Age at enrollment termasuk faktor terbesar penyumbang tingginya dropout.

______________________________________

📬 Contact
GitHub: @MSiswanto
Email: msiswanto@gmail.com
