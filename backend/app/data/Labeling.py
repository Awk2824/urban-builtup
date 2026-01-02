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

    if use_otsu:
        finite_endisi = endisi[valid_mask]
        if len(finite_endisi) == 0:
            raise RuntimeError("Tidak ada piksel valid untuk Otsu thresholding")
        threshold = threshold_otsu(finite_endisi)
        
    # Buat label (1 jika ENDISI > threshold)
    labels = np.zeros_like(endisi, dtype=int)
    labels[valid_mask & (endisi > threshold)] = 1

    # Masukkan label ke tabular
    new_tabular = np.hstack([tabular, labels.reshape(-1,1)])
    new_bands = bands + ["label_endisi"]

    return new_tabular, new_bands
#endregion