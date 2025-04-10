use linfa::prelude::*;
use linfa_svm::Svm;
use ndarray::{Array, Array2};
use csv::ReaderBuilder;
use std::{error::Error, io::Write, sync::{atomic::{AtomicBool, Ordering}, Arc}, time::Duration};
use std::thread;
use linfa_clustering::KMeans;
use plotters::prelude::*;

fn main() -> Result<(), Box<dyn Error>> {
    // Baca dataset dari string
    let data = r#"
no,tanah,penambahan_air_gram,penambahan_air_ml,kelembaban_manual,kelembaban_sensor,selisih
1,100gr,0 mL,0%,0.96%,0.96%,
2,100gr,10mL,10%,10.95%,0.95%,
3,100gr,20mL,20%,29.97%,9.97%,
4,100gr,30mL,30%,48.40%,18.40%,
5,100gr,40mL,40%,54.90%,14.90%,
6,100gr,50 mL,50%,71.00%,21.00%,
7,100gr,60mL,60%,77.70%,17.70%,
8,100gr,70mL,70%,77.98%,17.98%,
9,100gr,80mL,80%,82.54%,2.54%,
10,100gr,90mL,90%,85.35%,4.65%,
11,100gr,100mL,100%,85.45%,14.55%,
"#;

    let mut rdr = ReaderBuilder::new().from_reader(data.as_bytes());
    let mut records = Vec::new();

    println!("=== Data Asli ===");
    println!("No\tPenambahan Air (mL)\tKelembaban Sensor (%)");
    println!("------------------------------------------------");

    for (i, result) in rdr.records().enumerate() {
        let record = result?;
        let penambahan_air_ml: f64 = record[3].trim_end_matches('%').parse()?;
        let kelembaban_sensor: f64 = record[5].trim_end_matches('%').parse()?;
        println!("{}\t{}\t\t\t{}", i+1, penambahan_air_ml, kelembaban_sensor);
        records.push((penambahan_air_ml, kelembaban_sensor));
    }

    // Konversi ke Array2
    let mut features = Array2::zeros((records.len(), 1));
    let mut labels = Array::zeros(records.len());

    for (i, (penambahan_air_ml, kelembaban_sensor)) in records.iter().enumerate() {
        features[[i, 0]] = *penambahan_air_ml;
        labels[i] = *kelembaban_sensor;
    }

    // Bagi dataset menjadi training dan testing
    let (train, test) = linfa::dataset::Dataset::new(features, labels.clone()).split_with_ratio(0.8);

    // SVM
    println!("\n=== Training SVM ===");
    let svm = Svm::params()
        .gaussian_kernel(10.0)
        .fit(&train)?;

    let svm_pred = svm.predict(&test);
    
    println!("\n=== Hasil Prediksi SVM ===");
    println!("Data Test\tPrediksi");
    println!("--------------------------");
    for (i, pred) in svm_pred.iter().enumerate() {
        let x = test.records()[[i, 0]];
        println!("{:.1} mL\t\t{:.2}%", x, pred);
    }

    // K-Means
    println!("\n=== Training K-Means ===");
    let running_knn = Arc::new(AtomicBool::new(true));
    let running_knn_clone = running_knn.clone();

    let handle_knn = thread::spawn(move || {
        let mut dots = 0;
        while running_knn_clone.load(Ordering::Relaxed) {
            print!("\rTraining K-Means{}   ", ".".repeat(dots));
            dots = (dots + 1) % 4;
            std::io::stdout().flush().unwrap();
            thread::sleep(Duration::from_millis(500));
        }
        println!("\rTraining K-Means selesai!        ");
    });

    let knn_model = KMeans::params(3)
        .max_n_iterations(100)
        .fit(&train)?;

    running_knn.store(false, Ordering::Relaxed);
    handle_knn.join().unwrap();

    let knn_pred = knn_model.predict(&test);
    
    println!("\n=== Hasil Prediksi K-Means ===");
    println!("Data Test\tCluster");
    println!("--------------------------");
    for (i, &cluster) in knn_pred.iter().enumerate() {
        let x = test.records()[[i, 0]];
        println!("{:.1} mL\t\t{}", x, cluster);
    }

    // Buat grafik
    let root = BitMapBackend::new("plot.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Hasil Prediksi Model", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0f64..100f64, 0f64..100f64)?;

    chart.configure_mesh()
        .x_desc("Penambahan Air (mL)")
        .y_desc("Kelembaban Sensor (%)")
        .draw()?;

    // Plot data asli
    chart.draw_series(
        records.iter().map(|(x, y)| Circle::new((*x, *y), 5, BLUE.filled())),
    )?.label("Data Asli")
      .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.filled()));

    // Plot prediksi SVM
    chart.draw_series(
        test.records().outer_iter().zip(svm_pred.iter()).map(|(x, y)| {
            Circle::new((x[0], *y), 5, RED.filled())
        })
    )?.label("Prediksi SVM")
      .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.filled()));

    // Plot prediksi K-Means
    let cluster_colors = [GREEN, MAGENTA, CYAN];
    chart.draw_series(
        test.records().outer_iter().zip(knn_pred.iter()).map(|(x, &cluster)| {
            TriangleMarker::new((x[0], labels[x[0] as usize]), 10, cluster_colors[cluster].filled())
        })
    )?.label("Prediksi K-Means")
      .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], GREEN.filled()));

    chart.configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    println!("\nGrafik telah disimpan sebagai plot.png");

    Ok(())
}