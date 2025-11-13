"""
Tahapan:
1. Perhitungan Indeks ENDISI (Enhanced Normalized Difference Index Surface)
2. Pembentukan label ground truth (built-up / non-built-up)
"""
#region Import
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from skimage.filters import threshold_otsu

from app.config import Constants
#endregion


#region Labeling
def labeling(tabular: np.ndarray, bands: list[str], use_otsu: bool = True, threshold: float = Constants.ENDISI_THRESHOLD) -> Tuple[np.ndarray, list]:
    """
    - Band input: B02 (blue), B03 (green), B11 (swir1), B12 (swir2)
    - Menghitung ENDISI per piksel dan menambahkan kolom label (0/1)
    """

    # Cari index tiap band
    band_to_idx = {b: i for i, b in enumerate(bands)}
    required = ["B02","B03","B11","B12"]
    for b in required:
        if b not in band_to_idx:
            raise ValueError(f"Band diperlukan '{b}' tidak ditemukan di bands list: {bands}")

    # Ambil kolom sesuai urutan
    blue = tabular[:, band_to_idx["B02"]]
    green = tabular[:, band_to_idx["B03"]]
    swir1 = tabular[:, band_to_idx["B11"]]
    swir2 = tabular[:, band_to_idx["B12"]]

    # Mask valid: gunakan data yang valid saja
    valid_mask = np.isfinite(blue) & np.isfinite(green) & np.isfinite(swir1) & np.isfinite(swir2)

    # Hitung ENDISI
    mndwi = (green - swir1) / (green + swir1 + Constants.EPSILON)
    alpha = (2.0 * np.nanmean(blue[valid_mask])) / (np.nanmean(swir1[valid_mask]/(swir2[valid_mask]+Constants.EPSILON)) + np.nanmean(mndwi[valid_mask]**2) + Constants.EPSILON)
    endisi = (blue - alpha * ((swir1 / (swir2+Constants.EPSILON)) + mndwi**2)) / (blue + alpha * ((swir1 / (swir2+Constants.EPSILON)) + mndwi**2) + Constants.EPSILON)

    # # Plot ENDISI 2D sebelum flattening
    # plt.figure(figsize=(10,6))
    # plt.hist(endisi[valid_mask], bins=100, color='skyblue', edgecolor='black')
    # plt.title("Histogram ENDISI")
    # plt.xlabel("ENDISI")
    # plt.ylabel("Jumlah piksel")
    # plt.show()

    if use_otsu:
        finite_endisi = endisi[valid_mask]
        if len(finite_endisi) == 0:
            raise RuntimeError("Tidak ada piksel valid untuk Otsu thresholding")
        threshold = threshold_otsu(finite_endisi)
        # print(f"Threshold ENDISI menggunakan Otsu: {threshold:.4f}")

    # # Set invalid ke NaN
    # endisi[~valid_mask] = np.nan
    
    # # Buat label '1' jika ENDISI > threshold
    # labels = np.zeros(endisi.shape[0], dtype=np.uint8)
    # labels[np.isfinite(endisi) & (endisi > threshold)] = 1
    # labels = labels.astype(int)

    # unique, counts = np.unique(labels, return_counts=True)
    # print(f"Distribusi label ENDISI: {dict(zip(unique, counts))}")

    # # Masukkan label ke tabular
    # label_col = labels.reshape(-1, 1)
    # new_tabular = np.hstack([tabular, label_col])
    # new_bands = bands + ["label_endisi"]

    # print("\n=== Contoh data tabular dengan label ENDISI ===")
    # print("Kolom:", " ".join(new_bands))
    # np.set_printoptions(linewidth=200, precision=4, suppress=True)
    # print(new_tabular[:5])
    # print()


    # Buat label (1 jika ENDISI > threshold)
    labels = np.zeros_like(endisi, dtype=int)
    labels[valid_mask & (endisi > threshold)] = 1

    # Masukkan label ke tabular
    new_tabular = np.hstack([tabular, labels.reshape(-1,1)])
    new_bands = bands + ["label_endisi"]

    # print("=== Contoh 5 data tabular dengan label ENDISI ===")
    # print(np.round(new_tabular[:5], 4))

    return new_tabular, new_bands
#endregion