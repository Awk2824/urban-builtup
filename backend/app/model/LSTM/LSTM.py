import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import traceback
import json
import tempfile
from datetime import datetime

from app.config import Constants
from app.data.DatabaseManager import load_model_from_db, get_latest_sequence_for_wilayah


def generate_prediction_graph(history_dict, prediction_dict, output_path, history_dates, target_year):
    import datetime
    try:
        history_totals = [sum(history_dict[sub][i] for sub in Constants.SUBREGIONS) for i in range(len(history_dates))]

        # Pastikan semua tanggal adalah datetime
        pred_steps = list(prediction_dict.keys())
        pred_values = list(prediction_dict.values())

        plt.figure(figsize=(9, 5))
        plt.plot(history_dates, history_totals, marker='o', linestyle='-', label='Data Historis')

        if pred_values:
            plt.plot(
                [history_dates[-1]] + pred_steps,
                [history_totals[-1]] + pred_values,
                marker='x', linestyle='--', color='red', label=f'Prediksi hingga {target_year}'
            )

        plt.title(f'Prediksi Luas Lahan Terbangun hingga Tahun {target_year}')
        plt.xlabel('Tahun')
        plt.ylabel('Total Luas Lahan Terbangun (km²)')
        plt.legend()
        plt.grid(True)
        all_years = [d.year if isinstance(d, datetime.datetime) else d for d in history_dates] + pred_steps
        plt.xticks(all_years)
        plt.ylim(bottom=0)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return output_path
    except Exception as e:
        print(f"Gagal membuat grafik prediksi: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None



if __name__ == "__main__":
    prediction_graph_path = None
    try:
        if len(sys.argv) != 4:
            print("Error: Diperlukan 3 argumen: <result_output_path> <target_year> <id_wilayah>", file=sys.stderr)
            sys.exit(1)
        
        result_output_path = sys.argv[1]
        target_year = int(sys.argv[2])
        id_wilayah = int(sys.argv[3])

        model_data, model_id = load_model_from_db(model_type="LSTM")
        if model_data is None:
            raise RuntimeError("Gagal memuat model LSTM dari database.")

        model = model_data['model']
        scaler = model_data['scaler']
        sequence_length = model_data['sequence_length']

        print(f"\nMengambil {sequence_length} data historis terakhir untuk wilayah ID: {id_wilayah}...")
        sequences_dict, history_dates, last_date = get_latest_sequence_for_wilayah(id_wilayah, sequence_length)
        if sequences_dict is None:
            raise ValueError("Tidak dapat melanjutkan prediksi karena data historis sub-wilayah tidak cukup atau tidak lengkap.")
        
        history_for_graph = dict(sequences_dict)
        
        num_steps_to_predict = target_year - last_date
        print(f"\nMemulai prediksi sebanyak {num_steps_to_predict} langkah ke depan (berdasarkan urutan citra)...\n")

        # Persiapan data sequence
        current_sequences_np = {}
        for sub_wilayah in Constants.SUBREGIONS:
            current_sequences_np[sub_wilayah] = np.array(sequences_dict[sub_wilayah]).reshape(-1, 1)

        predictions = {}
    
        for step in range(num_steps_to_predict):
            total_luas_for_step = 0
            print(f"   - Prediksi langkah ke-{step + 1}")

            for sub_wilayah in Constants.SUBREGIONS:
                identifier = f"{id_wilayah}_{sub_wilayah}"
                current_scaler = scaler.get(identifier)
                
                if current_scaler is None:
                    # Gunakan skaler lain dari wilayah yang sama
                    fallback_keys = [k for k in scaler.keys() if k.startswith(f"{id_wilayah}_")]
                    if fallback_keys:
                        current_scaler = scaler[fallback_keys[0]]
                    else:
                        current_scaler = next(iter(scaler.values()))
    
                current_input_np = current_sequences_np[sub_wilayah]
                input_df = pd.DataFrame(current_input_np, columns=['area'])
                scaled_input = current_scaler.transform(input_df)
                reshaped_input = scaled_input.reshape(1, sequence_length, 1)
                scaled_prediction = model.predict(reshaped_input, verbose=0)
                prediction_scaled_df = pd.DataFrame(scaled_prediction, columns=['area'])
                prediction = current_scaler.inverse_transform(prediction_scaled_df)[0][0]
                last_history_value = current_input_np[-1][0]
                total_luas_for_step += prediction

                # Geser window sequence
                current_sequences_np[sub_wilayah] = np.vstack((current_input_np[1:], [[prediction]]))

            pred_year = last_date + step + 1
            predictions[pred_year] = total_luas_for_step
            print(f"   -> Prediksi Total Luas (langkah {step + 1}): {total_luas_for_step:.3f} km²")

        print("\n--- Prediksi Selesai ---")

        print("\nMembuat grafik prediksi...")
        pred_graph_temp_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        prediction_graph_path = generate_prediction_graph(history_for_graph, predictions, pred_graph_temp_path, history_dates, target_year)

        predicted_value_for_target_year = list(predictions.values())[-1] if predictions else None

        results = {
            "Predictions": predictions,
            "History": list(zip(history_dates, [sum(history_for_graph[sub][i] for sub in Constants.SUBREGIONS) for i in range(len(history_dates))])),
            "PredictionGraphPath": prediction_graph_path,
            "TargetYear": target_year,
            "PredictedValue": predicted_value_for_target_year
        }

        with open(result_output_path, 'w') as f:
            json.dump(results, f, default=float)

    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Terjadi kesalahan selama proses prediksi LSTM: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
    finally:
        print("\n--- Selesai ---")
