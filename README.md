## Gambaran Program
Program ini menunjukkan implementasi machine learning di Rust menggunakan crate Linfa untuk:
1. Support Vector Machine (SVM) untuk regresi
2. Clustering dengan K-Means
3. Visualisasi data dengan Plotters

## Langkah 1: Setup Proyek
1. Buat proyek baru:
```bash
cargo new rust-ml-example
cd rust-ml-example
```

2. Tambahkan dependensi ke `Cargo.toml`:
```toml
linfa-svm = "0.7.2"
linfa-nn = "0.7.1"
linfa-clustering = "0.7.1"
ndarray = "0.15.6"
csv = "1.1"
rand = "0.9.0"
plotters = "0.3.0"
```

## Langkah 2: Memahami Struktur Data
Program menggunakan dataset sederhana tentang:
- Penambahan air (mL)
- Kelembaban tanah (%)
Dataset dimasukkan langsung dalam kode sebagai string CSV.

```rust
let data = r#"
no,tanah,penambahan_air_gram,penambahan_air_ml,kelembaban_manual,kelembaban_sensor,selisih
1,100gr,0 mL,0%,0.96%,0.96%,
...
"#;
```

## Langkah 3: Persiapan Data
1. Baca data CSV:
```rust
let mut rdr = ReaderBuilder::new().from_reader(data.as_bytes());
```

2. Konversi ke format yang bisa diproses:
```rust
let mut features = Array2::zeros((records.len(), 1));
let mut labels = Array::zeros(records.len());
```

3. Bagi data menjadi training dan testing set:
```rust
let (train, test) = Dataset::new(features, labels).split_with_ratio(0.8);
```

## Langkah 4: Implementasi SVM
1. Buat model SVM:
```rust
let svm = Svm::params()
    .gaussian_kernel(10.0)
    .fit(&train)?;
```

2. Lakukan prediksi:
```rust
let svm_pred = svm.predict(&test);
```

## Langkah 5: Implementasi K-Means
1. Buat model K-Means dengan animasi loading:
```rust
let running_knn = Arc::new(AtomicBool::new(true));
// ... [code threading untuk animasi loading]

let knn_model = KMeans::params(3)
    .max_n_iterations(100)
    .fit(&train)?;
```

2. Lakukan prediksi clustering:
```rust
let knn_pred = knn_model.predict(&test);
```

## Langkah 6: Visualisasi Data
1. Setup canvas plot:
```rust
let root = BitMapBackend::new("plot.png", (800, 600)).into_drawing_area();
```

2. Buat chart:
```rust
let mut chart = ChartBuilder::on(&root)
    .caption("Hasil Prediksi Model", ("sans-serif", 30))
    .build_cartesian_2d(0f64..100f64, 0f64..100f64)?;
```

3. Plot data dan hasil prediksi:
```rust
chart.draw_series(...); // Data asli
chart.draw_series(...); // Prediksi SVM
chart.draw_series(...); // Prediksi K-Means
```

## Cara Menjalankan
1. Jalankan program:
```bash
cargo run
```

2. Output yang dihasilkan:
- Hasil prediksi SVM dan K-Means di terminal
- File gambar `plot.png` dengan visualisasi data

## Optimasi dan Pengembangan
1. **Gunakan Dataset Nyata**:
   - Baca dari file CSV eksternal
   - Gunakan dataset dari UCI Machine Learning Repository

2. **Evaluasi Model**:
   - Hitung metrik seperti MSE untuk SVM
   - Hitung silhouette score untuk K-Means

3. **Tambahkan Model Lain**:
   - Regresi Linear
   - Decision Trees
   - Neural Networks

## Contoh Output
```
=== Data Asli ===
No  Penambahan Air (mL)  Kelembaban Sensor (%)
------------------------------------------------
1   0                    0.96
2   10                   10.95
...

=== Hasil Prediksi SVM ===
Data Test    Prediksi
--------------------------
10.0 mL      12.34%
...

=== Hasil Prediksi K-Means ===
Data Test    Cluster
--------------------------
10.0 mL      1
...
```

Grafik akan disimpan sebagai `plot.png` dengan:
- Titik biru: Data asli
- Titik merah: Prediksi SVM
- Segitiga berwarna: Cluster K-Means

## Pembelajaran
Program ini menunjukkan:
1. Cara memproses data di Rust
2. Implementasi algoritma ML dasar
3. Visualisasi hasil
4. Threading untuk animasi loading

## Referensi
- [Dokumentasi Linfa](https://rust-ml.github.io/linfa/)
- [Contoh Plotters](https://plotters-rs.github.io/book/intro.html)
