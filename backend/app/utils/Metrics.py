"""
File ini berisikan kumpulan metrik yang digunakan untuk mengevaluasi model klasifikasi dan prediksi
Terdiri dari:
- Metrik untuk LSTM (MAE, MSE, RMSE)
- Metrik untuk Regresi Logistik (accuracy, precision, recall, f1-score)
"""

import math
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

#region Prediction
def prediction_metrics(y_true, y_pred, display = True):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    if display:
        print("=== Prediction Evaluation Metrics ===")
        print(f"MAE  : {mae:.4f}")
        print(f"MSE  : {mse:.4f}")
        print(f"RMSE : {rmse:.4f}")
        print(f"MAPE : {mape * 100:.2f}%")

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape
    }
#endregion


#region Classification
def classification_metrics(y_true, y_pred, display = True):
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    if display:
        print("\n=== Classification Evaluation Metrics ===")
        print(f"Confusion Matrix:")
        print(f"{cm}\n")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1-Score : {f1:.4f}")

    return {
        # "Confusion_Matrix": cm,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1Score": f1
    }
#endregion