# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

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

Setup environment:
```

```

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
Jelaskan cara menjalankan protoype sistem machine learning yang telah dibuat. Selain itu, sertakan juga link untuk mengakses prototype tersebut.

```

```

## Conclusion
Jelaskan konklusi dari proyek yang dikerjakan.
- Secara umum Curricular merupakan penyebab utama tingginya dropout, diikuti oleh Tuition fees up to date, Age at enrollment, Admission grade, Previous qualification dan Course.
- Umur saat sekolah (Age at enrollment) perlu dipertimbangkan karena termasuk faktor terbesar penyumbang tingginya dropout.
- Admission grade dan previous qualification juga berkorelasi positif dengan tingginya dropout.Perlu diperhatikan lebih siswa-siswa dengan admission-previous qualification grades yang rendah.

### Rekomendasi Action Items
Berikan beberapa rekomendasi action items yang harus dilakukan perusahaan guna menyelesaikan permasalahan atau mencapai target mereka.
- action item 1
- action item 2
