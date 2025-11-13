import os
import sys
import glob
import json
import numpy as np
import rasterio
import traceback
from PIL import Image, ImageDraw, ImageFont
from sklearn.preprocessing import StandardScaler

from app.data.Preprocessing import process_folder, raster_to_tabular, split_into_subregion
from app.model.LogisticRegression.Training import predict_labels
from app.config import Constants
from app.data.DatabaseManager import load_model_from_db, get_preprocessed_folder, log_classification_result, log_training_data, get_rgb_band_paths


#region Classification
def classification(input_folder, shapefile_path, weights, bias, preprocessing_output_folder, classified_output_folder, threshold=0.5):
    shapefile_name = os.path.splitext(os.path.basename(shapefile_path))[0]
    print("\n=== Mengecek Data Existing di Database ===")
    img_date_str = os.path.basename(os.path.normpath(input_folder))
    
    processed_folder = get_preprocessed_folder(img_date_str, shapefile_path)

    # Preprocessing citra baru jika tidak ada existing
    if processed_folder and os.path.exists(processed_folder):
        print(f"Data ditemukan di database. Menggunakan hasil preprocessing yang ada.")
    else:
        if processed_folder and not os.path.exists(processed_folder):
            print(f"Data ada di DB, tetapi path tidak valid di disk: {processed_folder}")
        
        print("Data tidak ditemukan. Memulai proses preprocessing citra baru...")
        processed_folder = process_folder(input_folder, shapefile_path, preprocessing_output_folder)
        print("Preprocessing selesai")
        
        if processed_folder:
            try:
                print("\n=== Mencatat data preprocessing baru ke database ===")
                log_training_data(
                    img_date_str = img_date_str,
                    shapefile_path = shapefile_path,
                    processed_folder_path = processed_folder
                )
                print("Pencatatan data preprocessing berhasil.")
            except Exception as log_ex:
                print(f"Gagal mencatat data preprocessing baru: {log_ex}", file=sys.stderr)
    
    print("\n=== Membagi Citra Menjadi Sub-Wilayah ===")
    subregion_folder = os.path.join(processed_folder, "sub-wilayah")
    clipped_files = glob.glob(os.path.join(processed_folder, f"{shapefile_name}_*.tif"))
    for file in clipped_files:
        band_name = os.path.basename(file).replace(f"{shapefile_name}_", "").replace(".tif", "")
        split_into_subregion(file, subregion_folder, shapefile_name, band_name)
    
    os.makedirs(classified_output_folder, exist_ok=True)
    subregion_outputs = {}
    area_summary = []
    total_builtup = 0
    total_nonbuilt = 0
    total_valid = 0
    pixel_area_km2 = (10 * 10) / 1000000  # 100 m2 = 0.0001 km2

    subregion_areas_dict = {}
    for subregion in Constants.SUBREGIONS:
        subregion_name = f"{shapefile_name}_{subregion}"
        try:
            tabular, _, combined_mask = raster_to_tabular(
                subregion_folder=subregion_folder,
                shapefile_name=shapefile_name,
                subregion_name=subregion_name,
                subregion=subregion
            )
        except Exception as e:
            print(f"Gagal memproses {subregion_name}: {e}")
            continue

        if tabular is None or len(tabular) == 0:
            print(f"Tidak ada data valid pada {subregion_name}")
            continue

        # Prediksi label biner
        X = tabular
        if Constants.NORMALIZE_FEATURES:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)  
            
        if X.shape[1] != len(weights):
            if X.shape[1] > len(weights):
                X = X[:, :len(weights)]
            else:
                raise ValueError("Jumlah fitur pada data lebih sedikit daripada yang dibutuhkan model.")
        labels = predict_labels(X, weights, bias, threshold=threshold)

        # Transformasi kembali ke raster
        raster_labels = np.zeros(combined_mask.shape, dtype=np.uint8)
        raster_labels[combined_mask] = labels

        # ======== Hitung luas lahan (km2) ========
        valid_pixels = combined_mask.sum()
        n_builtup = np.sum((raster_labels == 1) & combined_mask)
        n_nonbuilt = np.sum((raster_labels == 0) & combined_mask)

        total_valid += valid_pixels
        total_builtup += n_builtup
        total_nonbuilt += n_nonbuilt
        area_builtup_km2 = n_builtup * pixel_area_km2
        subregion_areas_dict[subregion] = area_builtup_km2
        area_nonbuilt_km2 = n_nonbuilt * pixel_area_km2

        print(f"\nLuas lahan terbangun: {area_builtup_km2:.3f} km2")
        print(f"Luas lahan non-terbangun: {area_nonbuilt_km2:.3f} km2")

        area_summary.append({
            "Subwilayah": subregion,
            "Built-up (km²)": round(area_builtup_km2, 3),
            "Non-built-up (km²)": round(area_nonbuilt_km2, 3)
        })

        rgb_array = np.zeros((3, raster_labels.shape[0], raster_labels.shape[1]), dtype=np.uint8)
        # # Built-up = merah
        # rgb_array[0][raster_labels == 1] = 139  # R
        # rgb_array[1][raster_labels == 1] = 0    # G
        # rgb_array[2][raster_labels == 1] = 0    # B
        # # Non-built-up = hijau
        # rgb_array[0][raster_labels == 0] = 0
        # rgb_array[1][raster_labels == 0] = 100
        # rgb_array[2][raster_labels == 0] = 0
        
        built = (raster_labels == 1) & combined_mask
        nonbuilt = (raster_labels == 0) & combined_mask
        
        rgb_array[:, :, :] = 0
        # Built-up = merah
        rgb_array[0][built] = 139
        rgb_array[1][built] = 0
        rgb_array[2][built] = 0

        # Non-built-up = hijau
        rgb_array[0][nonbuilt] = 0
        rgb_array[1][nonbuilt] = 100
        rgb_array[2][nonbuilt] = 0

        alpha_channel = (combined_mask.astype(np.uint8) * 255)
        rgba = np.dstack((rgb_array[0], rgb_array[1], rgb_array[2], alpha_channel))

        subregion_image = Image.fromarray(rgba, 'RGBA')
        subregion_outputs[subregion] = subregion_image
    
    total_pixels = total_builtup + total_nonbuilt   
    area_total_km2 = total_valid * pixel_area_km2
    area_builtup_total_km2 = total_builtup * pixel_area_km2
    area_nonbuilt_total_km2 = total_nonbuilt * pixel_area_km2
    print(f"\nTotal area raster: {total_pixels} pixels")
    print(f"\nTotal area raster valid: {area_total_km2:.3f} km2")
    print(f"Total lahan terbangun: {area_builtup_total_km2:.3f} km2")
    print(f"Total lahan non-terbangun: {area_nonbuilt_total_km2:.3f} km2")

    print("\n--- Membuat raster kecamatan utuh ---")

    # b4_path, b3_path, b2_path = get_rgb_band_paths(img_date_str, shapefile_path)
    b4_path = os.path.join(processed_folder, f"{shapefile_name}_B04.tif")
    b3_path = os.path.join(processed_folder, f"{shapefile_name}_B03.tif")
    b2_path = os.path.join(processed_folder, f"{shapefile_name}_B02.tif")
    tcc_image_path = None
    if b4_path and b3_path and b2_path:
        tcc_output_path = os.path.join(classified_output_folder, f"TCC_{shapefile_name}.png")
        os.makedirs(os.path.dirname(tcc_output_path), exist_ok=True)
        tcc_image_path = generate_tcc(b4_path, b3_path, b2_path, tcc_output_path)
    else:
        print("- Gagal mendapatkan path band B2/B3/B4 dari database. Citra asli tidak dapat dibuat.")
        
    # Pastikan keempat sub-wilayah tersedia
    required_regions = Constants.SUBREGIONS
    if not all(r in subregion_outputs for r in required_regions):
        print("Tidak semua sub-wilayah tersedia. Citra tidak dapat digabung.")
        return 0.0, 0.0, None, subregion_areas_dict, tcc_image_path

    # Baca tiap gambar (sub-wilayah)
    img_NW = subregion_outputs["NW"]
    img_NE = subregion_outputs["NE"]
    img_SW = subregion_outputs["SW"]
    img_SE = subregion_outputs["SE"]

    # Dapatkan ukuran (harus sama)
    w, h = img_NW.size

    # Buat kanvas kosong 2x2
    mosaic = Image.new('RGBA', (w*2, h*2), (0, 0, 0, 0))

    # Tempelkan tiap subregion di posisi spasial yang benar
    mosaic.paste(img_NW, (0, 0))
    mosaic.paste(img_NE, (w, 0))
    mosaic.paste(img_SW, (0, h))
    mosaic.paste(img_SE, (w, h))
    
    legend_height = 60
    final_image = Image.new('RGBA', (mosaic.width, mosaic.height + legend_height), (255, 255, 255, 255))
    final_image.paste(mosaic, (0, 0))
    
    draw = ImageDraw.Draw(final_image)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    # Legenda Lahan Terbangun
    draw.rectangle([20, mosaic.height + 15, 40, mosaic.height + 35], fill=(139, 0, 0))
    draw.text((50, mosaic.height + 20), "Lahan Terbangun", fill="black", font=font)
    
    # Legenda Lahan Non-Terbangun
    draw.rectangle([220, mosaic.height + 15, 240, mosaic.height + 35], fill=(0, 100, 0))
    draw.text((250, mosaic.height + 20), "Lahan Non-Terbangun", fill="black", font=font)

    output_mosaic_path = os.path.join(classified_output_folder, f"{shapefile_name}.png")
    mosaic.save(output_mosaic_path)

    print(f"Hasil visualisasi disimpan di: {output_mosaic_path}")
    
    classified_image_path = output_mosaic_path

    output_overlay_path = os.path.join(
        classified_output_folder,
        f"{shapefile_name}_overlay.png"
    )
    
    generate_overlay(
        tcc_path=tcc_image_path,
        mosaic_path=classified_image_path,
        output_overlay_path=output_overlay_path,
        alpha=100
    )
    print("\n=== PROSES KLASIFIKASI SELESAI ===")
    
    return area_builtup_total_km2, area_nonbuilt_total_km2, output_mosaic_path, subregion_areas_dict, tcc_image_path, output_overlay_path
#endregion


#region Generate Image
def generate_tcc(b4_path: str, b3_path: str, b2_path: str, output_tcc_path: str):
    try:
        nodata_values = set()
        profile = None
        
        with rasterio.open(b4_path) as r:
            red = r.read(1)
            nodata = r.nodata
            profile = r.profile
            if nodata is not None: nodata_values.add(nodata)
        with rasterio.open(b3_path) as g:
            green = g.read(1)
            nodata = g.nodata
            if nodata is not None: nodata_values.add(nodata)
        with rasterio.open(b2_path) as b:
            blue = b.read(1)
            nodata = b.nodata
            if nodata is not None: nodata_values.add(nodata)

        # Stack menjadi satu array (bands, height, width)
        rgb = np.stack([red, green, blue], axis=0).astype(np.float32)

        # Hapus nilai NoData jika ada
        valid_mask = np.ones(red.shape, dtype=bool)
        valid_mask &= (red != 0) & (green != 0) & (blue != 0)

        for ndv in nodata_values:
            valid_mask &= (red != ndv) & (green != ndv) & (blue != ndv)
        
        alpha_channel = (valid_mask * 255).astype(np.uint8)
        
        rgb[~np.isfinite(rgb)] = 0

        # Contrast stretching sederhana (persentil 2-98)
        stretched_rgb = np.zeros_like(rgb, dtype=np.uint8)
        for i in range(3):
            valid_pixels_band = rgb[i, :, :][valid_mask]
            if len(valid_pixels_band) > 1:
                try:
                    p2, p98 = np.percentile(valid_pixels_band, (2, 98))
                    denominator = (p98 - p2) if (p98 - p2) > 1e-6 else 1.0
                    band_stretched = np.clip((rgb[i, :, :] - p2) * 255.0 / denominator, 0, 255)
                    stretched_rgb[i, :, :] = band_stretched.astype(np.uint8)
                except IndexError:
                     stretched_rgb[i, :, :] = np.clip(rgb[i, :, :], 0, 255).astype(np.uint8)

            elif len(valid_pixels_band) == 1:
                 stretched_rgb[i, :, :][valid_mask] = 0


        # Ubah urutan menjadi (height, width, bands) untuk PIL
        rgb_image_array = np.moveaxis(stretched_rgb, 0, -1)
        rgba_image_array = np.dstack((rgb_image_array, alpha_channel))

        img = Image.fromarray(rgba_image_array, 'RGBA')
        img.save(output_tcc_path)
        return output_tcc_path

    except Exception as e:
        print(f"Gagal membuat citra asli: {e}", file=sys.stderr)
        return None


def generate_overlay(tcc_path, mosaic_path, output_overlay_path, alpha=120):
    """
    Membuat overlay antara citra asli (TCC) dan citra hasil klasifikasi
    """

    # Baca kedua gambar
    tcc = Image.open(tcc_path).convert("RGBA")
    cls = Image.open(mosaic_path).convert("RGBA")

    min_w = min(tcc.width, cls.width)
    min_h = min(tcc.height, cls.height)

    tcc = tcc.crop((0, 0, min_w, min_h))
    cls = cls.crop((0, 0, min_w, min_h))
    
    # Atur transparansi layer klasifikasi
    r, g, b, a = cls.split()
    new_alpha = a.point(lambda px: alpha if px > 0 else 0)
    cls = Image.merge("RGBA", (r, g, b, new_alpha))

    # Composite
    overlay = Image.alpha_composite(tcc, cls)
    overlay.save(output_overlay_path)

    return output_overlay_path
#endregion

    
#region Main
if __name__ == "__main__":
    # sys.argv akan berisi:
    # [0] -> "path/ke/LogReg.py"
    # [1] -> input_folder
    # [2] -> shapefile_path
    # [3] -> output_folder
    # [4] -> area_output_path
    
    if len(sys.argv) != 5:
        print("Error: Diperlukan 4 argumen: <input_folder> <shapefile_path> <output_folder> <area_output_path>", file=sys.stderr)
        sys.exit(1)

    input_folder_arg = sys.argv[1]
    shapefile_path_arg = sys.argv[2]
    output_folder_arg = sys.argv[3]
    area_output_path_arg = sys.argv[4]
    
    try:
        shapefile_name = os.path.splitext(os.path.basename(shapefile_path_arg))[0]
        
        project_data_folder = os.path.dirname(output_folder_arg)
        img_date_str_for_folder = os.path.basename(os.path.normpath(input_folder_arg))
        classified_output_path_arg = os.path.join(project_data_folder, "Classified", img_date_str_for_folder)
        
        print("=== Memulai Proses Klasifikasi Lahan Terbangun ===")
        
        model_data, latest_model_id = load_model_from_db(model_type="LogReg")
        
        if model_data is None:
            raise RuntimeError(f"Gagal memuat model regresi logistik dari database.")

        weights = model_data.get("weights")
        bias = model_data.get("bias")
        threshold = model_data.get("threshold")

        if weights is None or bias is None:
            raise ValueError("Objek model yang dimuat dari database tidak memiliki 'weights' atau 'bias'.")
        
        built_up_area, non_built_up_area, final_image_path, subregion_areas, tcc_path, output_overlay_path = classification(
                                                                                                                input_folder = input_folder_arg,
                                                                                                                shapefile_path = shapefile_path_arg,
                                                                                                                weights = weights,
                                                                                                                bias = bias,
                                                                                                                preprocessing_output_folder = output_folder_arg,
                                                                                                                classified_output_folder = classified_output_path_arg,
                                                                                                                threshold = threshold
                                                                                                            )
        
        img_date_str = os.path.basename(os.path.normpath(input_folder_arg))
        log_classification_result(
            nama_kec = shapefile_name,
            img_date_str = img_date_str,
            model_id = latest_model_id,
            subregion_areas = subregion_areas
        )
        
        results = {
            "BuiltUpArea": built_up_area,
            "NonBuiltUpArea": non_built_up_area,
            "ImagePath": final_image_path,
            "RealImagePath": tcc_path,
            "OverlayPath": output_overlay_path
        }
        
        with open(area_output_path_arg, 'w') as f:
            json.dump(results, f)

    except Exception as e:
        print(f"Terjadi kesalahan fatal selama proses klasifikasi: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        
    finally:
        print("\n--- Selesai ---")
        # input("Tekan Enter untuk keluar...")
#endregion