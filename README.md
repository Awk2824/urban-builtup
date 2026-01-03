# TerraBuild

TerraBuild adalah aplikasi desktop berbasis Windows yang digunakan untuk melakukan klasifikasi dan prediksi lahan terbangun menggunakan citra satelit Sentinel-2.

Aplikasi mengintegrasikan:
- Frontend desktop berbasis **C# .NET (WPF)**
- Backend pemrosesan berbasis **Python**
- Database lokal **PostgreSQL**
- Model Machine Learning (Regresi Logistik dan LSTM)


## Fitur Utama

- Klasifikasi area lahan terbangun dan non-terbangun
- Prediksi luas lahan terbangun berbasis data time series
- Pengolahan citra satelit Sentinel-2
- Antarmuka desktop interaktif
- Backend Python terintegrasi secara internal (Python embedded)


## Panduan Instalasi 

1. Klik tombol **Code → Download ZIP** pada repository ini
2. Ekstrak file ZIP ke folder lokal

### Instalasi Aplikasi

Untuk mengunduh setup installer (.exe) aplikasi, klik [disini](https://github.com/Awk2824/urban-builtup/releases/tag/v1.0)


### Instalasi Database

- Aplikasi ini menggunakan database PostgreSQL agar dapat berfungsi dengan baik. 
- Panduan lengkap instalasi database dan proses restore skema database tersedia dalam `UserManual_535220004.pdf`
- File skema database dalam diakses di `Installation/Database/Schema_535220004.sql`


##

Panduan lengkap penggunaan aplikasi, termasuk:
- Instalasi aplikasi
- Instalasi database
- Alur penggunaan sistem

dapat diakses melalui file `UserManual_535220004.pdf`
