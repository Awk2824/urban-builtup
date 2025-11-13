"""
Script ini hanya untuk mengekspor shapefile per kecamatan
Tidak berkaitan dengan proses pada aplikasi
Data batas wilayah dapat diunduh di https://tanahair.indonesia.go.id/portal-web/unduh
"""


import geopandas as gpd
import fiona
import os

# Path ke folder .gdb
gdb_path = "D:/Copernicus/RBI10K_ADMINISTRASI_DESA_20230928.gdb"

# # List semua layer (untuk pengecekan)
# layers = fiona.listlayers(gdb_path)
# print("Layers:", layers)

# Output folder untuk shapefile per kecamatan
output_dir = "D:/Copernicus/Shapefile"
os.makedirs(output_dir, exist_ok=True)

# Load layer
gdf = gpd.read_file(gdb_path, layer="ADMINISTRASI_AR_DESAKEL") # nama layer disesuaikan
# print(gdf.columns) # untuk melihat index nama kolom pada file geodatabase
# print(gdf["WADMKC"].unique()) # untuk melihat list kecamatan pada file geodatabase

# Daftar kecamatan yang ingin diambil shapefilenya
kecamatan_tangerang = [
    "Batuceper", "Benda", "Cibodas", "Ciledug", "Cipondoh", "Jatiuwung",
    "Karang Tengah", "Karawaci", "Larangan", "Neglasari", "Periuk",
    "Pinang", "Tangerang"
]

kecamatan_tangsel = [
    "Ciputat", "Ciputat Timur", "Pamulang", "Pondok Aren",
    "Serpong", "Serpong Utara", "Setu"
]

# Pisahkan ke dalam subfolder (opsional)
tangerang_dir = os.path.join(output_dir, "Kota_Tangerang")
tangsel_dir = os.path.join(output_dir, "Kota_Tangerang_Selatan")
os.makedirs(tangerang_dir, exist_ok=True)
os.makedirs(tangsel_dir, exist_ok=True)

# Simpan shapefile per kecamatan
def save_kecamatan(kecamatan, folder):
    for name in kecamatan:
        if name in gdf["WADMKC"].unique():
            print(f"Kecamatan {name} ditemukan!")

            subset = gdf[gdf["WADMKC"] == name]
            out_path = os.path.join(folder, f"{name.replace(' ', '_')}.shp")
            subset.to_file(out_path)
        else:
            print(f"Kecamatan {name} tidak ditemukan!")


# Jalankan program
save_kecamatan(kecamatan_tangerang, tangerang_dir)
save_kecamatan(kecamatan_tangsel, tangsel_dir)