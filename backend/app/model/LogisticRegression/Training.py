import numpy as np
import os
import sys
import time
import json
import traceback
import re
from typing import Tuple
from collections import defaultdict
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app.config import Constants
from app.utils import Metrics
from app.data.Preprocessing import split_into_subregion, raster_to_tabular
from app.data.Labeling import labeling
from app.data.DatabaseManager import get_training_data, save_model_to_db

#region Initialization
def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def initialize_weights(n_features: int) -> Tuple[np.ndarray, float]:
    weights = np.zeros(n_features)
    bias = 0.0
    return weights, bias
#endregion


#region Compute Loss
def compute_loss(y_true, y_pred, weights=None, penalty="l2", C=1.0):
    m = len(y_true)
    y_pred = np.clip(y_pred, Constants.EPSILON, 1 - Constants.EPSILON)
    loss = -(1 / m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    reg_term = 0
    if weights is not None:
        if penalty == "l2":
            reg_term = (1 / (2 * C)) * np.sum(weights ** 2)
        elif penalty == "l1":
            reg_term = (1 / C) * np.sum(np.abs(weights))
        
    return loss + reg_term
#endregion


#region Threshold
def find_optimal_threshold(X, y_true, weights, bias):
    probabilities = sigmoid(np.dot(X, weights) + bias)
    thresholds = np.linspace(0, 1, 101)
    best_threshold, best_f1, best_metrics = 0.5, -1, None

    for t in thresholds:
        preds = (probabilities >= t).astype(int)
        metrics = Metrics.classification_metrics(y_true, preds, display=False)
        if metrics["F1Score"] > best_f1:
            best_threshold, best_f1, best_metrics = t, metrics["F1Score"], metrics

    return best_threshold, best_metrics
#endregion


#region Train Model
def train_logistic_regression(X, y, lr=0.01, epochs=1000, penalty="l2", C=1.0, verbose=False):
    m, n = X.shape
    weights, bias = initialize_weights(n)
    losses = []

    classes = np.unique(y)
    class_weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weight_dict = {c: w for c, w in zip(classes, class_weights)}

    for epoch in range(epochs):
        # Prediksi probabilitas
        linear_model = np.dot(X, weights) + bias
        y_pred = sigmoid(linear_model)

        # Hitung loss
        loss = compute_loss(y, y_pred, weights=weights, penalty=penalty, C=C)
        losses.append(loss)
            
        # # Gradient descent
        sample_weights = np.array([class_weight_dict[label] for label in y])
        dw = (1/m) * np.dot(X.T, sample_weights * (y_pred - y))
        db = (1/m) * np.sum(sample_weights * (y_pred - y))

        if penalty == "l2":
            dw += (1 / C) * weights
        elif penalty == "l1":
            dw += (1 / C) * np.sign(weights)
            
        # Update bobot dan bias
        weights -= lr * dw
        bias -= lr * db

        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss:.9f}")

        # Berhenti jika perubahan loss sangat kecil
        if epoch > 0 and abs(losses[-2] - losses[-1]) < Constants.EPSILON:
            print(f"Konvergen pada epoch ke-{epoch}")
            break

    return weights, bias, losses


def predict_labels(X, weights, bias, threshold=0.5):
    probabilities = sigmoid(np.dot(X, weights) + bias)
    return (probabilities >= threshold).astype(int)
#endregion
    
#region Main
if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            print("Error: Diperlukan 1 argumen: <path_ke_file_metrik_output>", file=sys.stderr)
            sys.exit(1)
        metrics_output_path = sys.argv[1]
        
        print("=== Mengambil Data dari Database ===")
        all_clipped_files = get_training_data()

        if not all_clipped_files:
            print("Error: Tidak ada data pelatihan yang ditemukan di database.", file=sys.stderr)
            sys.exit(1)

        # Verifikasi file berada dalam lokal
        existing_files = [path for path in all_clipped_files if os.path.exists(path)]
        if len(existing_files) != len(all_clipped_files):
             print("Peringatan: Beberapa file yang tercatat di database tidak ditemukan di disk.", file=sys.stderr)
        
        if not existing_files:
            print("Error: Tidak ada file valid yang ditemukan di disk berdasarkan path dari database.", file=sys.stderr)
            sys.exit(1)
            
        print(f"Ditemukan {len(existing_files)} total path band dari database.")
        
        print("\n=== Mengelompokkan Data Berdasarkan Tanggal dan Kecamatan ===")
        grouped_images = defaultdict(list)
        for path in existing_files:
            match = re.search(r'(\d{4}_\d{2}_\d{2}[\\/][^\\/]+)', path)
            if match:
                image_key = match.group(1)
                grouped_images[image_key].append(path)
        
        print(f"Ditemukan {len(grouped_images)} citra unik untuk diproses.")
        
        all_labeled_data = []
        
        for image_key, files in grouped_images.items():
            print(f"\n--- Memproses citra: {image_key} ---")
            
            processed_folder = os.path.dirname(files[0])
            shapefile_name = os.path.basename(image_key).split('\\')[-1].split('/')[-1]

            print("- Membagi citra menjadi sub-wilayah...")
            subregion_folder = os.path.join(processed_folder, "sub-wilayah")
            for file in files:
                band_name_full = os.path.basename(file)
                band_name = band_name_full.replace(f"{shapefile_name}_", "").replace(".tif", "")
                split_into_subregion(file, subregion_folder, shapefile_name, band_name)

            print("- Melakukan labeling...")
            for subregion in Constants.SUBREGIONS:
                subregion_name = f"{shapefile_name}_{subregion}"
                tabular, bands, _ = raster_to_tabular(subregion_folder, shapefile_name, subregion_name, subregion)
                
                if tabular is not None and len(tabular) > 0:
                    new_tabular, _ = labeling(tabular, bands)
                    all_labeled_data.append(new_tabular)
                else:
                    print(f"Tidak ada data valid pada {subregion_name}.")

        if not all_labeled_data:
            raise RuntimeError("Tidak ada data yang valid setelah semua proses labeling.")
        
        dataset = np.vstack(all_labeled_data)
        # print(f"Dataset gabungan berhasil dibuat: {dataset.shape[0]:,} sampel")

        print("\n=== Memulai Proses Pelatihan Model Regresi Logistik ===")
        start = time.time()
        X = dataset[:, :-1]
        y = dataset[:, -1]
        
        if Constants.NORMALIZE_FEATURES:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        lr = 0.02
        penalty = "l2"
        c = 100.0
        weights, bias, losses = train_logistic_regression(X_train, y_train, lr=lr, epochs=1500, penalty=penalty, C=c, verbose=True)
        
        print("\nMencari threshold optimal...\n")
        best_threshold, best_metrics = find_optimal_threshold(X_test, y_test, weights, bias)
        
        y_pred = predict_labels(X_test, weights, bias, threshold=best_threshold)
        
        print("\n=== Evaluasi Akhir Model ===")
        final_metrics = Metrics.classification_metrics(y_test, y_pred)
        
        # Simpan model yang sudah dilatih
        print("\nMenyimpan model regresi logistik...")
        
        model_data_to_save = {
            "weights": weights,
            "bias": bias,
            "losses": losses,
            "threshold": best_threshold,
            "lr": lr,
            "penalty": penalty,
            "C": c
        }
        
        save_model_to_db(model_data_to_save, "LogReg", best_metrics)
        
        with open(metrics_output_path, 'w') as f:
            for key, value in best_metrics.items():
                if isinstance(value, np.generic):
                    best_metrics[key] = value.item()
                elif isinstance(value, np.ndarray):
                    best_metrics[key] = value.tolist()
            json.dump(best_metrics, f)
            
        print("\n--- Proses Pelatihan Selesai ---")
        end = time.time()
        total_time = end - start
        print(f"Total waktu pelatihan: {total_time:.2f} detik")
        
    except Exception as e:
        print(f"Terjadi kesalahan: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        
    finally:
        print("\n--- Selesai ---")
        # input("Tekan Enter untuk keluar...")
#endregion