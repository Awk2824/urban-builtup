"""
Tahapan:
 1. Membaca citra Sentinel-2
 2. Resampling citra ke resolusi 10m
 3. Pemotongan berdasarkan shapefile
 4. Pembagian hasil potongan menjadi 4 sub-wilayah (NW, NE, SE, SW)
 5. Konversi data raster ke dalam bentuk tabular
"""
#region Import
import os
import sys
import traceback
import glob
import rasterio
import geopandas as gpd
import numpy as np
from rasterio.mask import mask
from rasterio.enums import Resampling
from rasterio.warp import reproject
from shapely.geometry import mapping, box
from tqdm import tqdm

from app.config import Constants
from app.data.DatabaseManager import log_training_data
#endregion


#region Resampling
def resample_raster(band_path, resolution = Constants.TARGET_RESOLUTION):
    with rasterio.open(band_path) as src:
        src_res = src.res[0]

        if abs(src_res - resolution) < 1e-3:
            data = src.read()
            profile = src.profile.copy()
            return data, profile

        scale = src_res / resolution
        new_height = int(src.height * scale)
        new_width = int(src.width * scale)

        data = src.read(
            out_shape = (src.count, new_height, new_width),
            resampling = Resampling.bilinear
        )

        transform = src.transform * src.transform.scale(
            src.width / data.shape[-1],
            src.height / data.shape[-2]
        )

        profile = src.profile.copy()
        profile.update({
            "height": data.shape[1],
            "width": data.shape[2],
            "transform": transform,
            "crs": src.crs
        })

        return data, profile
#endregion


#region Clipping
def clip_raster(raster_path, shapefile_path, output_path):
    shp = gpd.read_file(shapefile_path)
    with rasterio.open(raster_path) as src:
        shp = shp.to_crs(src.crs)
        geoms = [mapping(geom) for geom in shp.geometry]
        out_image, out_transform = mask(src, geoms, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)


def split_into_subregion(raster_path, output_folder, shapefile_name, band_name):
    os.makedirs(output_folder, exist_ok=True)

    with rasterio.open(raster_path) as src:
        data = src.read(1)
        profile = src.profile

        height, width = data.shape
        # half_h = height // 2
        # half_w = width // 2
        top_h = height // 2
        bottom_h = height - top_h

        left_w = width // 2
        right_w = width - left_w

        # Bounding box per sub-wilayah
        # quadrants = {
        #     "NW": data[0:half_h, 0:half_w],                     # North West
        #     "NE": data[0:half_h, half_w:width],                 # North East
        #     "SE": data[half_h:height, half_w:width],            # South East
        #     "SW": data[half_h:height, 0:half_w]                 # South West
        # }
        quadrants = {
            "NW": data[0:top_h, 0:left_w],
            "NE": data[0:top_h, left_w:width],
            "SW": data[top_h:height, 0:left_w],
            "SE": data[top_h:height, left_w:width]
        }

        for name, geom in quadrants.items():
            # if name == "NW":
            #     row_off, col_off = 0, 0
            # elif name == "NE":
            #     row_off, col_off = 0, half_w
            # elif name == "SW":
            #     row_off, col_off = half_h, 0
            # else:
            #     row_off, col_off = half_h, half_w
            if name == "NW":
                row_off, col_off = 0, 0
            elif name == "NE":
                row_off, col_off = 0, left_w
            elif name == "SW":
                row_off, col_off = top_h, 0
            else:
                row_off, col_off = top_h, left_w                

            new_transform = src.transform * rasterio.Affine.translation(col_off, row_off)

            new_meta = profile.copy()
            new_meta.update({
                "height": geom.shape[0],
                "width": geom.shape[1],
                "transform": new_transform
            })

            # Buat nama sub-wilayah
            output_path = os.path.join(output_folder, f"{shapefile_name}_{band_name}_{name}.tif")

            with rasterio.open(output_path, "w", **new_meta) as dest:
                dest.write(geom, 1)

            # print(f"Sub-wilayah {shapefile_name}_{band_name}_{name} berhasil dibuat")
#endregion


#region Konversi ke Tabular
def raster_to_tabular(subregion_folder, shapefile_name, subregion_name, subregion):
    tif_files = sorted(glob.glob(os.path.join(subregion_folder, f"{shapefile_name}_*_{subregion}.tif")))

    if not tif_files:
        print(f"Tidak ditemukan file raster di {subregion_folder}")
        return

    band_data = {}
    nodata_mask = None

    min_height, min_width = None, None
    rasters = {}

    # Baca semua raster
    for tif in tif_files:
        with rasterio.open(tif) as src:
            data = src.read(1).astype(np.float32)
            rasters[tif] = data

            if min_height is None or data.shape[0] < min_height:
                min_height = data.shape[0]
            if min_width is None or data.shape[1] < min_width:
                min_width = data.shape[1]

    combined_mask = np.ones((min_height, min_width), dtype=bool)

    # Potong semua raster agar ukurannya seragam
    for tif, data in rasters.items():
        trimmed = data[:min_height, :min_width]

        with rasterio.open(tif) as src:
            nodata = src.nodata

        # Mask valid piksel
        mask_valid = np.ones_like(trimmed, dtype=bool)
        if nodata is not None:
            mask_valid &= trimmed != nodata
        mask_valid &= trimmed != 0

        # Gabungkan mask
        combined_mask &= mask_valid

        # Ambil nama band dari filename
        filename = os.path.basename(tif)
        parts = filename.split("_")
        band_name = parts[-2] if len(parts) > 2 else os.path.splitext(filename)[0]

        band_data[band_name] = trimmed

    for b in band_data:
        band_data[b] = band_data[b][combined_mask].flatten() * Constants.SCALE_FACTOR

    lengths = [len(v) for v in band_data.values()]
    if len(set(lengths)) > 1:
        raise ValueError(f"Jumlah piksel tiap band tidak sama: {dict(zip(band_data.keys(), lengths))}")

    # Gabungkan jadi data tabular
    bands = sorted(band_data.keys())
    tabular = np.column_stack([band_data[b] for b in bands])

    # num_pixels = len(next(iter(band_data.values())))
    # print(f"\n=== Contoh data tabular untuk sub-wilayah {subregion_name} ({num_pixels} piksel) ===")
    # print("Kolom:", " ".join(bands))
    # np.set_printoptions(linewidth=200)
    # print(tabular[:5])
    # print()

    return tabular, bands, combined_mask
#endregion


#region Preprocessing
# Menyamakan grid raster ke raster referensi
def align_to_reference(src, ref_profile):
    data = np.empty((src.count, ref_profile["height"], ref_profile["width"]), dtype=src.dtypes[0])
    reproject(
        source=rasterio.band(src, 1),
        destination=data[0],
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=ref_profile["transform"],
        dst_crs=ref_profile["crs"],
        resampling=Resampling.bilinear
    )
    return data


def process_folder(folder_path, shapefile_path, output):
    jp2_files = sorted(glob.glob(os.path.join(folder_path, "*.jp2")))

    if not jp2_files:
        raise FileNotFoundError(f"Format file tidak sesuai pada folder {folder_path}")

    # Baca shapefile
    shp = gpd.read_file(shapefile_path)
    shapefile_name = os.path.splitext(os.path.basename(shapefile_path))[0]

    # Nama folder berdasarkan tanggal citra
    date_name = os.path.basename(os.path.normpath(folder_path)) 

    # Buat folder output per tanggal per kecamatan
    output_date_folder = os.path.join(output, date_name, shapefile_name)
    os.makedirs(output_date_folder, exist_ok=True)

    # Gunakan band B04 (Red) sebagai referensi (karena resolusinya 10m dan digunakan untuk RGB)
    ref_band = next((f for f in jp2_files if "B04" in os.path.basename(f)), jp2_files[0])
    with rasterio.open(ref_band) as ref_src:
        ref_profile = ref_src.profile
        ref_data = ref_src.read()
    # print(f"Band referensi: {os.path.basename(ref_band)}")

    for jp2_file in tqdm(jp2_files, desc=f"Proses folder {date_name} - {shapefile_name} "):
        # Ambil nama band
        band_name = os.path.splitext(os.path.basename(jp2_file))[0]
        
        if band_name.endswith(("_10m", "_20m", "_60m")):
            band_name = band_name.rsplit("_", 1)[0]

        # Pengecekan agar band 10m tidak diresampling
        if "_20m" in jp2_file:
            # Hanya band 20m yang diresample ke 10m
            data, profile = resample_raster(jp2_file, resolution=Constants.TARGET_RESOLUTION)
        elif "_10m" in jp2_file:
            # Band 10m langsung dibaca tanpa resampling
            with rasterio.open(jp2_file) as src:
                data = src.read()
                profile = src.profile
        # print(f"\nResampling {shapefile_name}_{band_name} berhasil")

        # Selaraskan band ke grid referensi
        with rasterio.io.MemoryFile() as mem:
            with mem.open(**profile) as tmp_src:
                tmp_src.write(data)
                aligned_data = align_to_reference(tmp_src, ref_profile)

        profile = ref_profile.copy()
        profile.update({
            "count": aligned_data.shape[0],
            "dtype": aligned_data.dtype
        })
        # print(f"Align {shapefile_name}_{band_name} berhasil")

        # Clipping berdasarkan shapefile
        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(**profile) as tmp:
                tmp.write(aligned_data)

                shp = shp.to_crs(tmp.crs)
                geoms = [mapping(geom) for geom in shp.geometry]
                out_image, out_transform = mask(tmp, geoms, crop=True)
                out_meta = tmp.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform
                })

        # Simpan hasil clip berdasarkan shapefile
        clipped_filename = f"{shapefile_name}_{band_name}.tif"
        clipped_path = os.path.join(output_date_folder, clipped_filename)

        with rasterio.open(clipped_path, "w", **out_meta) as dest:
            dest.write(out_image)
        # print(f"Clipping {shapefile_name}_{band_name} berhasil")

        # # Bagi kecamatan menjadi 4 sub-wilayah (NW, NE, SE, SW)
        # subregion_folder = os.path.join(output_date_folder, "sub-wilayah")
        # split_into_subregion(clipped_path, subregion_folder, shapefile_name, band_name)
        # print(f"Pembagian {shapefile_name} menjadi 4 sub-wilayah berhasil\n")
    
    # # Konversi dari raster ke bentuk tabular
    # print("===== Konversi tiap sub-wilayah ke bentuk tabular =====")
    # for subregion in Constants.SUBREGIONS:
    #     print(f"Konversi wilayah {shapefile_name}_{subregion} ke bentuk tabular")
    #     subregion_name = f"{shapefile_name}_{subregion}"
    #     subregion_path = os.path.join(output_date_folder, "sub-wilayah")
    #     raster_to_tabular(subregion_path, shapefile_name, subregion_name, subregion)

    # print(f"Semua band dari {folder_path} berhasil dilakukan preprocessing")
    # print()
    return output_date_folder
#endregion


#region Main
if __name__ == "__main__":
    # sys.argv akan berisi:
    # [0] -> "path/ke/Preprocessing.py"
    # [1] -> folder_path
    # [2] -> shapefile_path
    # [3] -> output_path

    if len(sys.argv) != 4:
        print("Error: Diperlukan 3 parameter: <folder_path> <shapefile_path> <output_path>")
        sys.exit(1)

    folder_path_arg = sys.argv[1]
    shapefile_path_arg = sys.argv[2]
    output_path_arg = sys.argv[3]

    print("=== Memulai preprocessing citra Sentinel-2 ===")
    print(f"Folder Input: {folder_path_arg}")
    print(f"Shapefile: {shapefile_path_arg}")
    print(f"Folder Output: {output_path_arg}")
    print()

    try:
        processed_folder = process_folder(
            folder_path=folder_path_arg,
            shapefile_path=shapefile_path_arg,
            output=output_path_arg
        )
        print(f"\nProses pemotongan citra berhasil.")
        
        print("\n=== Memulai penyimpanan data ke database ===")
        img_date_str = os.path.basename(os.path.normpath(folder_path_arg))
        log_training_data(
            img_date_str=img_date_str,
            shapefile_path=shapefile_path_arg,
            processed_folder_path=processed_folder
        )
        print(f"\nProses berhasil. Hasil tersimpan di: {processed_folder}")
        
    except Exception as e:
        print(f"Terjadi kesalahan: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        
    finally:
        print("\n--- Selesai ---")
        # input("Tekan Enter untuk keluar...")
#endregion