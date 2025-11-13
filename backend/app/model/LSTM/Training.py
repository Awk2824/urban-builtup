import numpy as np
import sys
import re
import os
import json
import base64
import time
import pandas as pd
import tempfile
import traceback
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow import keras

from app.data.DatabaseManager import get_all_training_paths, log_area_data_for_lstm, get_lstm_training_data, save_model_to_db
from app.data.Preprocessing import split_into_subregion, raster_to_tabular
from app.data.Labeling import labeling
from app.config import Constants
from app.utils import Metrics



#region Cleaning Data
def clean_and_prepare_series(raw_data, seq_length):
    """
    Membersihkan data (deteksi outlier, interpolasi, ambil maksimum per tahun),
    serta identifier pada LSTM
    """
    all_X, all_y, all_identifiers = [], [], []
    scalers = {}

    for (id_wilayah, sub_wilayah), records in raw_data.items():
        if len(records) <= seq_length:
            continue

        # Konversi ke DataFrame
        df = pd.DataFrame(records, columns=['date', 'area'])
        df['year'] = df['date'].apply(lambda d: d.year)

        # Ambil maksimum per tahun
        df = df.groupby('year', as_index=False)['area'].max().sort_values('year')

        # Deteksi dan perbaiki outlier (Robust Z-Score menggunakan MAD)
        median = df['area'].median()
        
        # Hitung Median Absolute Deviation (MAD)
        if median == 0:
             mad = np.median(np.abs(df['area'] - median).dropna())
        else:
             mad = np.median(np.abs(df['area'] - median).dropna())

        if mad > 0:
            z_score_threshold = 2.5 # Ambang batas yang umum untuk MAD
            # Hitung modified z-score
            df['mod_z_score'] = 0.6745 * (df['area'] - median) / mad
            df['area'] = df['area'].apply(lambda x: np.nan if abs(0.6745 * (x - median) / mad) > z_score_threshold else x)
        
        # Interpolasi nilai NaN yang baru dibuat (atau yang sudah ada)
        df['area'] = df['area'].interpolate(method='linear')

        df = df.dropna(subset=['area'])
        if df.empty or df['area'].isna().any() or df['area'].eq(0).all() or len(df) <= seq_length:
            continue

        # Tambah identifier
        identifier = f"{id_wilayah}_{sub_wilayah}"

        # Normalisasi per identifier
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[['area']])
        scalers[identifier] = scaler

        # Bentuk sequence untuk LSTM
        X_seq, y_seq = create_sequences(scaled, seq_length)
        if len(X_seq) > 0:
            all_X.append(X_seq)
            all_y.append(y_seq)
            # Lacak identifier untuk setiap sequence
            all_identifiers.extend([identifier] * len(X_seq))

    if not all_X:
        raise ValueError("Tidak ada data valid untuk pelatihan")

    X_full = np.vstack(all_X)
    y_full = np.vstack(all_y)
    identifiers_full = np.array(all_identifiers)
    
    # Kembalikan identifiers_full dan scalers (dictionary)
    return X_full, y_full, identifiers_full, scalers
#endregion


#region Generate Data
def generate_data_lstm_per_year():
    success_count = 0
    fail_count = 0
    try:
        print("Mengambil data dari database...")
        all_clipped_files_info = get_all_training_paths(include_metadata=True)
        if not all_clipped_files_info:
            raise ValueError("Tidak ada data pelatihan yang ditemukan.")

        # group per (id_wilayah, year)
        grouped_by_year = defaultdict(list)  

        for info in all_clipped_files_info:
            img_date = info.get('img_date')
            id_wilayah = info.get('id_wilayah')
            year = img_date.year
            grouped_by_year[(id_wilayah, year)].append(info)

        for (id_wilayah, year), images in grouped_by_year.items():
            print(f"\nMemproses wilayah dengan ID: {id_wilayah} pada tahun {year}...")

            max_total_area = 0.0
            best_image_data = None

            # Evaluasi semua citra di tahun yang sama
            for info in images:
                try:
                    img_date = info['img_date']
                    shapefile_path = info['path_shp']
                    band_paths = {k: v for k, v in info.items()
                                  if k.endswith('_path') and k != 'path_shp' and v and os.path.exists(v)}
                    if not {'b2_path', 'b3_path', 'b11_path', 'b12_path'}.issubset(band_paths.keys()):
                        print(f"- {img_date}: Band tidak lengkap.")
                        continue

                    shapefile_name = os.path.splitext(os.path.basename(shapefile_path))[0]
                    processed_folder = os.path.dirname(list(band_paths.values())[0])
                    subregion_folder = os.path.join(processed_folder, "sub-wilayah")
                    os.makedirs(subregion_folder, exist_ok=True)

                    # Split tiap band ke sub-wilayah
                    for band_key, band_path in band_paths.items():
                        base = band_key.replace('_path', '')
                        m = re.match(r'[bB](\d+)', base)
                        band_name_simple = f'B{int(m.group(1)):02d}' if m else base.upper()
                        split_into_subregion(band_path, subregion_folder, shapefile_name, band_name_simple)

                    pixel_area_km2 = (10 * 10) / 1_000_000
                    subregion_areas_endisi = {}

                    # Hitung area built-up untuk tiap sub-wilayah
                    for subregion in Constants.SUBREGIONS:
                        subregion_name = f"{shapefile_name}_{subregion}"
                        try:
                            tabular, bands, _ = raster_to_tabular(
                                subregion_folder=subregion_folder,
                                shapefile_name=shapefile_name,
                                subregion_name=subregion_name,
                                subregion=subregion
                            )
                            if tabular is None or len(tabular) == 0:
                                continue

                            labeled_tabular, _ = labeling(tabular, bands, use_otsu=True)
                            labels = labeled_tabular[:, -1]
                            n_builtup_endisi = np.sum(labels == 1)
                            area_builtup_km2 = n_builtup_endisi * pixel_area_km2
                            subregion_areas_endisi[subregion] = area_builtup_km2

                        except Exception as e:
                            print(f"Error {subregion_name}: {e}", file=sys.stderr)
                            fail_count += 1

                    # Hitung total area built-up semua subregion
                    total_area = sum(subregion_areas_endisi.values())

                    # Simpan yang memiliki total area terbesar
                    if total_area > max_total_area:
                        max_total_area = total_area
                        best_image_data = {
                            "id_wilayah": id_wilayah,
                            "year": year,
                            "img_date": img_date,
                            "areas": subregion_areas_endisi
                        }

                except Exception as e:
                    print(f"Kesalahan memproses citra tahun {year}: {e}", file=sys.stderr)
                    fail_count += 1
                    continue

            # Jika citra terbaik ditemukan untuk tahun tersebut -> simpan ke DB
            if best_image_data is not None:
                log_area_data_for_lstm(
                    best_image_data["id_wilayah"],
                    best_image_data["img_date"],
                    best_image_data["areas"]
                )
                success_count += 1
                # print(f"Tahun {year}: citra {best_image_data['img_date']} dipilih (total area={max_total_area:.2f} km²)")
            else:
                print(f"Tidak ada citra valid untuk tahun {year}.")
                fail_count += 1

        if success_count == 0 and (fail_count > 0 or len(grouped_by_year) == 0):
            raise RuntimeError("Gagal menghasilkan data. Tidak bisa melanjutkan pelatihan.")
        return True

    except Exception as e:
        print(f"Terjadi kesalahan: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return False
#endregion


def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

        
#region Generate R2 Graph
class R2History(keras.callbacks.Callback):
    def __init__(self, X_val, y_val, val_identifiers, scalers_dict):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.val_identifiers = val_identifiers
        self.scalers_dict = scalers_dict 
        self.r2_scores = []

    def on_epoch_end(self, epoch, logs=None):
        try:
            all_y_pred_denorm = []
            all_y_val_denorm = []
            unique_val_ids = np.unique(self.val_identifiers)
            
            # Loop untuk setiap identifier unik di set validasi
            for identifier in unique_val_ids:
                indices = np.where(self.val_identifiers == identifier)[0]
                if len(indices) == 0:
                    continue
                    
                X_val_id = self.X_val[indices]
                y_val_id = self.y_val[indices]
                
                scaler = self.scalers_dict.get(identifier)
                if scaler is None:
                    continue
                    
                y_pred_id = self.model.predict(X_val_id, verbose=0)
                all_y_pred_denorm.append(scaler.inverse_transform(y_pred_id))
                all_y_val_denorm.append(scaler.inverse_transform(y_val_id))

            if not all_y_val_denorm:
                print(f"Epoch {epoch+1}: R² = N/A (Scaler lookup failed)")
                self.r2_scores.append(np.nan)
                return

            # Gabungkan semua hasil denormalisasi
            y_pred_denorm = np.vstack(all_y_pred_denorm)
            y_val_denorm = np.vstack(all_y_val_denorm)
            
            # Hitung R2 gabungan
            r2 = r2_score(y_val_denorm, y_pred_denorm)
            self.r2_scores.append(r2)
            print(f"Epoch {epoch+1}: R² = {r2:.4f}")
        
        except Exception as e:
            print(f"Error in R2History callback: {e}", file=sys.stderr)
            self.r2_scores.append(np.nan)
        
        
def generate_r2_graph(y_true_denorm, y_pred_denorm, output_path, years = None):
    try:
        y_true = np.array(y_true_denorm).flatten()
        y_pred = np.array(y_pred_denorm).flatten()
        
        if years is None:
            x = np.arange(1, len(y_true) + 1)
            x_label = "Sampel"
        else:
            x = np.array(years)
            x_label = "Tahun"

        plt.figure(figsize=(8, 5))
        plt.plot(x, y_true, 'o-', label='Actual (ground truth)', linewidth=2)
        plt.plot(x, y_pred, 'x--', label='Predicted (model)', linewidth=2)
        plt.xlabel(x_label)
        plt.ylabel('Luas Terbangun (km²)')
        plt.title('Perbandingan Nilai Aktual vs Prediksi')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.ylim(bottom=0)
        if years is not None:
            plt.xticks(np.arange(min(years), max(years) + 1, 1))
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return output_path
    except Exception as e:
        print(f"Gagal membuat grafik: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None
#endregion


#region Main
if __name__ == "__main__":
    r2_graph_path = None
    try:
        if len(sys.argv) != 9: # 1 (script) + 1 (metrics file) + 7 (hyperparams)
            print("Error: Diperlukan 8 argumen: <metrics_output_path> <seq_length> <epochs> <learning_rate> <hidden_units> <batch_size> <dropout_rate> <optimizer>", file=sys.stderr)
            sys.exit(1)
            
        print("\n=== Memulai Pelatihan Model ===")
        
        metrics_output_path = sys.argv[1]
        SEQ_LENGTH = int(sys.argv[2])
        EPOCHS = int(sys.argv[3])
        LEARNING_RATE = float(sys.argv[4])
        HIDDEN_UNITS = int(sys.argv[5])
        BATCH_SIZE = int(sys.argv[6])
        DROPOUT_RATE = float(sys.argv[7])
        OPTIMIZER_NAME = sys.argv[8].lower()
             
        data_generation_success = generate_data_lstm_per_year()
        if not data_generation_success:
            sys.exit(1)
             
        raw_data = get_lstm_training_data()
        if not raw_data:
            raise ValueError("Tidak ada data yang berhasil diambil dari database.")
        
        expanded_data = raw_data
                
        # Cleaning + Normalisasi per identifier
        X_full, y_full, identifiers_full, scalers = clean_and_prepare_series(expanded_data, SEQ_LENGTH)

        X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
            X_full, y_full, identifiers_full, 
            test_size=0.2, 
            random_state=42, 
            shuffle=False
        )
        
        print("\nMembangun model LSTM...")
        start = time.time()
        model = keras.models.Sequential(
            [
                keras.layers.LSTM(units = HIDDEN_UNITS, return_sequences = True, input_shape = (SEQ_LENGTH, 1)),
                keras.layers.Dropout(DROPOUT_RATE),
                keras.layers.LSTM(units = HIDDEN_UNITS, return_sequences = False),
                keras.layers.Dropout(DROPOUT_RATE),
                keras.layers.Dense(units = 1, activation = 'relu')
            ]
        )
        
        if OPTIMIZER_NAME == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        elif OPTIMIZER_NAME == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)
        elif OPTIMIZER_NAME == 'sgd':
            optimizer = keras.optimizers.SGD(learning_rate=LEARNING_RATE)
        else:
            print(f"Optimizer '{OPTIMIZER_NAME}' tidak dikenal. Menggunakan Adam sebagai default.", file=sys.stderr)
            optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
            
        model.compile(optimizer = optimizer, loss = 'mean_squared_error')
        model.summary()
        
        print(f"\nMemulai pelatihan model LSTM...")
        print(f"Hyperparameters: Epochs={EPOCHS}, Batch Size={BATCH_SIZE}, LR={LEARNING_RATE}, Hidden Units={HIDDEN_UNITS}, Dropout={DROPOUT_RATE}, Seq Len={SEQ_LENGTH}")
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        r2_callback = R2History(X_test, y_test, id_test, scalers)
        model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test), verbose = 2, callbacks = [r2_callback, early_stop])
        
        print("Pelatihan model LSTM selesai.")

        print("\nMengevaluasi model pada data pengujian...")
        all_y_pred_denorm = []
        all_y_test_denorm = []
        unique_test_ids = np.unique(id_test)
        
        for identifier in unique_test_ids:
            indices = np.where(id_test == identifier)[0]
            if len(indices) == 0:
                continue
                
            X_test_id = X_test[indices]
            y_test_id = y_test[indices]
            
            scaler = scalers.get(identifier)
            if scaler is None:
                continue
                
            predictions_scaled_id = model.predict(X_test_id, verbose=0)
            y_pred_denorm_id = scaler.inverse_transform(predictions_scaled_id)
            y_test_denorm_id = scaler.inverse_transform(y_test_id)
            
            all_y_pred_denorm.append(y_pred_denorm_id)
            all_y_test_denorm.append(y_test_denorm_id)
            
        if not all_y_test_denorm:
            raise ValueError("Evaluasi gagal, tidak ada data uji yang berhasil didenormalisasi.")
            
        # Gabungkan semua hasil denormalisasi
        y_pred_denorm = np.vstack(all_y_pred_denorm)
        y_test_denorm = np.vstack(all_y_test_denorm)
        
        r2_graph_temp_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        generate_r2_graph(y_test_denorm, y_pred_denorm, r2_graph_temp_path)

        final_metrics = Metrics.prediction_metrics(y_test_denorm, y_pred_denorm)
        
        y_true_flat = y_test_denorm.flatten()
        y_pred_flat = y_pred_denorm.flatten()
        r2 = r2_score(y_true_flat, y_pred_flat)
        final_metrics["R2"] = r2
        final_metrics["R2GraphPath"] = r2_graph_temp_path
        final_metrics["R2PerEpoch"] = r2_callback.r2_scores
        
        model_data_to_save = {
            'model': model,
            'scaler': scalers,
            'sequence_length': SEQ_LENGTH,
            'optimizer': OPTIMIZER_NAME,
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'hidden_units': HIDDEN_UNITS,
            'dropout': DROPOUT_RATE
        }
        
        save_model_to_db(model_data_to_save, "LSTM", final_metrics)
        
        for key, value in final_metrics.items():
            if isinstance(value, np.generic): 
                final_metrics[key] = value.item()
        
            
        if not final_metrics.get("R2GraphPath"):
            print("Peringatan:1 Tidak ada data grafik R² yang tersimpan dalam metrics.")

        # Simpan ke file JSON untuk dikirim ke frontend
        with open(metrics_output_path, 'w') as f:
            json.dump(final_metrics, f)
            
            
        if not final_metrics.get("R2GraphPath"):
            print("Peringatan:2 Tidak ada data grafik R² yang tersimpan dalam metrics.")
            
        print("\n--- Proses Pelatihan Selesai ---")
        end = time.time()
        total_time = end - start
        print(f"Total waktu pelatihan: {total_time:.2f} detik")

    except Exception as e:
        print(f"Terjadi kesalahan fatal selama proses training LSTM: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
    finally:
        print("\n--- Selesai ---")
        # input("Tekan Enter untuk keluar...")
#endregion