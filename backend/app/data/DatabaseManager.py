import os
import sys
import psycopg2
import json
import io
import re
import tempfile
import base64
from joblib import dump, load
from datetime import datetime
from collections import defaultdict
from tensorflow import keras

from app.config import Constants

#region Log Training Data
def log_training_data(img_date_str, shapefile_path, processed_folder_path):
    conn = None
    try:
        print("\nMenghubungkan dengan database...\n")
        # Menghubungkan ke database menggunakan konfigurasi dari Constants
        conn = psycopg2.connect(**Constants.DB_CONFIG)
        cursor = conn.cursor()

        nama_kec = os.path.splitext(os.path.basename(shapefile_path))[0]

        cursor.execute("SELECT id_wilayah FROM wilayah_administrasi WHERE nama_kec = %s", (nama_kec,))
        result = cursor.fetchone()

        if result:
            id_wilayah, existing_path = result
            print(f"Wilayah '{nama_kec}' ditemukan dengan ID {id_wilayah}.")

            # Update path_shp jika berbeda atau kosong
            if existing_path != shapefile_path:
                print(f"Memperbarui path untuk wilayah '{nama_kec}'...")
                cursor.execute(
                    "UPDATE wilayah_administrasi SET path_shp = %s WHERE id_wilayah = %s",
                    (shapefile_path, id_wilayah)
                )
        else:
            print(f"Wilayah '{nama_kec}' tidak ditemukan, memasukkan data baru...")
            cursor.execute(
                "INSERT INTO wilayah_administrasi (nama_kec, path_shp) VALUES (%s, %s) RETURNING id_wilayah",
                (nama_kec, shapefile_path)
            )
            id_wilayah = cursor.fetchone()[0]
            print(f"Wilayah '{nama_kec}' berhasil ditambahkan dengan ID: {id_wilayah}")

        img_date = datetime.strptime(img_date_str, '%Y_%m_%d').strftime('%Y-%m-%d')
        
        band_paths = {}
        band_names = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B11', 'B12']
        for band in band_names:
            # Contoh path: D:\Skripsi\OutputData\2018_06_19\Batuceper\Batuceper_B02.tif
            file_path = os.path.join(processed_folder_path, f"{nama_kec}_{band}.tif")
            column_name = f"b{band.replace('B0', '').replace('B', '')}_path"
            
            # Simpan path jika file ada, jika tidak simpan None
            band_paths[column_name] = file_path if os.path.exists(file_path) else None

        # Cek duplikat
        cursor.execute(
            "SELECT id_train FROM pelatihan WHERE img_date = %s AND id_wilayah = %s",
            (img_date, id_wilayah)
        )
        existing_training = cursor.fetchone()

        # Jika ada, update data
        if existing_training:
            print(f"Data pelatihan untuk {nama_kec} pada tanggal {img_date} sudah ada. Memperbarui path...")
            update_query = """
                UPDATE pelatihan SET
                    b2_path = %(b2_path)s, b3_path = %(b3_path)s, b4_path = %(b4_path)s,
                    b5_path = %(b5_path)s, b6_path = %(b6_path)s, b7_path = %(b7_path)s,
                    b8_path = %(b8_path)s, b11_path = %(b11_path)s, b12_path = %(b12_path)s
                WHERE id_train = %(id_train)s
            """
            cursor.execute(update_query, {**band_paths, 'id_train': existing_training[0]})

        else:
            print(f"Memasukkan data pelatihan baru untuk {nama_kec} pada tanggal {img_date}...")
            columns = ", ".join(band_paths.keys())
            values_placeholder = ", ".join(["%s"] * len(band_paths))
            
            insert_query = f"""
                INSERT INTO pelatihan (img_date, id_wilayah, {columns})
                VALUES (%s, %s, {values_placeholder})
            """
            values = [img_date, id_wilayah] + list(band_paths.values())
            cursor.execute(insert_query, values)

        conn.commit()
        print("Data berhasil disimpan ke database.")

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error from database: {error}", file=sys.stderr)
        if conn:
            conn.rollback()

    finally:
        if conn is not None:
            conn.close()
#endregion


#region Get Training Data
def get_training_data():
    conn = None
    try:
        print("\nMenghubungkan dengan database...\n")
        conn = psycopg2.connect(**Constants.DB_CONFIG)
        cursor = conn.cursor()
        
        query = """
            SELECT p.b2_path, p.b3_path, p.b4_path, p.b5_path, p.b6_path,
                   p.b7_path, p.b8_path, p.b11_path, p.b12_path
            FROM pelatihan p;
        """
        cursor.execute(query)
        results = cursor.fetchall()

        if not results:
            print("Tidak ada data pelatihan yang ditemukan di database.", file=sys.stderr)
            return None

        all_paths = []
        for row in results:
            all_paths.extend([path for path in row if path is not None])
            
        return all_paths

    except psycopg2.Error as e:
        print(f"Error from database: {e}", file=sys.stderr)
        return None
    finally:
        if conn:
            conn.close()
            

def get_all_training_paths(include_metadata=False):
    conn = None
    all_paths = []
    all_info = []
    try:
        conn = psycopg2.connect(**Constants.DB_CONFIG)
        cursor = conn.cursor()
        query = """
            SELECT
                p.b2_path, p.b3_path, p.b4_path, p.b5_path, p.b6_path,
                p.b7_path, p.b8_path, p.b11_path, p.b12_path,
                p.id_wilayah, p.img_date, w.path_shp, w.nama_kec
            FROM pelatihan p
            JOIN wilayah_administrasi w ON p.id_wilayah = w.id_wilayah;
        """
        cursor.execute(query)
        records = cursor.fetchall()
        
        if not records:
             print("Tidak ada data pelatihan yang ditemukan di database.", file=sys.stderr)
             return [] if include_metadata else None 

        colnames = [desc[0] for desc in cursor.description]

        for record in records:
            record_dict = dict(zip(colnames, record))
            paths_in_record = [p for k, p in record_dict.items() if k.endswith('_path') and k != 'path_shp' and p]
            valid_paths = [p for p in paths_in_record if os.path.exists(p)]

            if include_metadata:
                 if valid_paths:
                    all_info.append(record_dict)
            else:
                all_paths.extend(valid_paths)

        if include_metadata:
            return all_info
        else:
            return all_paths

    except psycopg2.Error as e:
        print(f"Error from database: {e}", file=sys.stderr)
        return None
    finally:
        if conn: 
            conn.close()
        
            
            
def get_lstm_training_data():
    conn = None
    try:
        print("\nMenghubungkan dengan database...\n")
        conn = psycopg2.connect(**Constants.DB_CONFIG)
        cursor = conn.cursor()
        
        print("\nMengambil data deret waktu untuk pelatihan LSTM...")
        # Query untuk mengambil semua data, diurutkan berdasarkan tanggal
        query = """
            SELECT id_wilayah, sub_wilayah, img_date, luas_km2
            FROM luas_terbangun
            ORDER BY id_wilayah, sub_wilayah, img_date;
        """
        cursor.execute(query)
        
        # Mengelompokkan hasil ke dalam dictionary
        time_series_data = defaultdict(list)
        for id_wilayah, sub_wilayah, img_date, luas_km2 in cursor.fetchall():
            time_series_data[(id_wilayah, sub_wilayah)].append((img_date, luas_km2))

        return dict(time_series_data)

    except psycopg2.Error as e:
        print(f"Error from database: {e}", file=sys.stderr)
        return None
    finally:
        if conn:
            conn.close()
#endregion


#region Get RGB Band Path
def get_rgb_band_paths(img_date_str: str, shapefile_path: str):
    conn = None
    try:
        print("\nMenghubungkan dengan database...\n")
        conn = psycopg2.connect(**Constants.DB_CONFIG)
        cursor = conn.cursor()
        nama_kec = os.path.splitext(os.path.basename(shapefile_path))[0]
        img_date = datetime.strptime(img_date_str, "%Y_%m_%d").date()

        query = """
            SELECT p.b4_path, p.b3_path, p.b2_path
            FROM pelatihan p
            JOIN wilayah_administrasi w ON p.id_wilayah = w.id_wilayah
            WHERE w.nama_kec = %s AND p.img_date = %s;
        """
        cursor.execute(query, (nama_kec, img_date))
        record = cursor.fetchone()

        if record:
            if all(record):
                return record[0], record[1], record[2]
            else:
                 print(f"Peringatan: Tidak semua path B2/B3/B4 ditemukan di DB untuk {nama_kec} - {img_date_str}", file=sys.stderr)
                 return None, None, None
        else:
            return None, None, None

    except (psycopg2.Error, ValueError) as e:
        print(f"Error from database: {e}", file=sys.stderr)
        return None, None, None
    finally:
        if conn:
            conn.close()
#endregion


#region Check Existing Data
def get_preprocessed_folder(img_date_str: str, shapefile_path: str):
    """
    Memeriksa apakah data preprocessing sudah ada untuk tanggal dan shapefile tertentu.
    Jika ada, kembalikan path ke folder yang sudah diproses
    """
    conn = None
    try:
        print("\nMenghubungkan dengan database...\n")
        conn = psycopg2.connect(**Constants.DB_CONFIG)
        cursor = conn.cursor()

        nama_kec = os.path.splitext(os.path.basename(shapefile_path))[0]

        img_date = datetime.strptime(img_date_str, "%Y_%m_%d").date()

        query = """
            SELECT p.b2_path
            FROM pelatihan p
            JOIN wilayah_administrasi w ON p.id_wilayah = w.id_wilayah
            WHERE w.nama_kec = %s AND p.img_date = %s;
        """
        cursor.execute(query, (nama_kec, img_date))
        record = cursor.fetchone()

        if record and record[0]:
            return os.path.dirname(record[0])
        else:
            return None

    except (psycopg2.Error, ValueError) as e:
        print(f"Error from database: {e}", file=sys.stderr)
        return None
    finally:
        if conn:
            conn.close()
#endregion
            

#region Log Classification Result
def log_classification_result(nama_kec: str, img_date_str: str, model_id: int, subregion_areas: dict):
    """
    Mencatat hasil proses klasifikasi ke dalam tabel 'pengujian' dan 'luas_terbangun'
    """
    conn = None
    try:
        print("\nMenghubungkan dengan database...\n")
        conn = psycopg2.connect(**Constants.DB_CONFIG)
        cursor = conn.cursor()

        # Dapatkan id_wilayah dari nama_kec
        cursor.execute("SELECT id_wilayah FROM wilayah_administrasi WHERE nama_kec = %s;", (nama_kec,))
        wilayah_record = cursor.fetchone()
        if not wilayah_record:
            raise ValueError(f"Wilayah '{nama_kec}' tidak ditemukan di database.")
        id_wilayah = wilayah_record[0]

        # Konversi string tanggal ke objek tanggal
        img_date = datetime.strptime(img_date_str, "%Y_%m_%d").date()

        # Insert ke tabel pengujian
        print(f" - Mencatat pengujian untuk wilayah ID: {id_wilayah}, model ID: {model_id}...")
        test_query = "INSERT INTO pengujian (test_date, id_wilayah, img_date, id_model) VALUES (NOW(), %s, %s, %s);"
        cursor.execute(test_query, (id_wilayah, img_date, model_id))

        conn.commit()
        print("Pencatatan hasil klasifikasi berhasil")

    except (psycopg2.Error, ValueError) as e:
        print(f"Error from database: {e}", file=sys.stderr)
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()
#endregion


#region Log Data LSTM
def log_area_data_for_lstm(id_wilayah: int, img_date: datetime.date, subregion_areas: dict):
    conn = None
    try:
        conn = psycopg2.connect(**Constants.DB_CONFIG)
        cursor = conn.cursor()

        for sub_wilayah, luas_km2 in subregion_areas.items():
            # Cek apakah sudah ada
            check_query = "SELECT id_luas FROM luas_terbangun WHERE id_wilayah = %s AND img_date = %s AND sub_wilayah = %s;"
            cursor.execute(check_query, (id_wilayah, img_date, sub_wilayah))
            existing_record = cursor.fetchone()

            if existing_record:
                # Update
                update_query = "UPDATE luas_terbangun SET luas_km2 = %s WHERE id_luas = %s;"
                cursor.execute(update_query, (float(luas_km2), existing_record[0]))
            else:
                # Insert
                insert_query = "INSERT INTO luas_terbangun (id_wilayah, img_date, sub_wilayah, luas_km2) VALUES (%s, %s, %s, %s);"
                cursor.execute(insert_query, (id_wilayah, img_date, sub_wilayah, float(luas_km2)))

        conn.commit()

    except (psycopg2.Error, ValueError) as e:
        print(f"Error from database: {e}", file=sys.stderr)
        if conn: 
            conn.rollback()
    finally:
        if conn: 
            conn.close()
#endregion


#region Latest Sequence
def get_latest_sequence_for_wilayah(id_wilayah: int, sequence_length: int):
    conn = None
    try:
        print("\nMenghubungkan dengan database...\n")
        conn = psycopg2.connect(**Constants.DB_CONFIG)
        cursor = conn.cursor()

        query = """
            SELECT img_date, sub_wilayah, luas_terbangun
            FROM luas_terbangun
            WHERE id_wilayah = %s
            ORDER BY img_date ASC;
        """
        cursor.execute(query, (id_wilayah,))
        records = cursor.fetchall()

        if not records:
            print("Tidak ada data luas terbangun untuk wilayah ini.")
            return None

        data_dict = {sub: {} for sub in Constants.SUBREGIONS}
        for row in records:
            img_date, sub, luas = row
            try:
                if luas is None:
                    val = 0.0
                elif isinstance(luas, (float, int)):
                    val = float(luas)
                elif isinstance(luas, (bytes, bytearray)):
                    s = luas.decode('utf-8', errors='ignore').strip()
                    matches = re.findall(r"[-+]?\d*\.\d+|\d+", s)
                    val = float(matches[-1]) if matches else 0.0
                elif isinstance(luas, str):
                    s = luas.strip()
                    # jika format tuple-like "(..., ..., value)"
                    if s.startswith("(") and s.endswith(")"):
                        inner = s[1:-1].strip()
                        parts = inner.rsplit(",", maxsplit=1)
                        candidate = parts[-1].strip()
                        # hapus quotes/spasi
                        candidate = candidate.strip("'\" ")
                        try:
                            val = float(candidate)
                        except Exception:
                            matches = re.findall(r"[-+]?\d*\.\d+|\d+", s)
                            val = float(matches[-1]) if matches else 0.0
                    else:
                        matches = re.findall(r"[-+]?\d*\.\d+|\d+", s)
                        val = float(matches[-1]) if matches else 0.0
                else:
                    val = float(luas)
            except Exception:
                print(f"Warning: gagal parse luas_terbangun: {luas}", file=sys.stderr)
                val = 0.0

            year = img_date.year
            if sub in data_dict:
                # simpan hanya tanggal terbaru untuk tahun tersebut
                if year not in data_dict[sub] or img_date > data_dict[sub][year][0]:
                    data_dict[sub][year] = (img_date, val)

        year_sorted = sorted({y for sub in data_dict.values() for y in sub.keys()})
        
        trimmed_dict = {sub: [] for sub in Constants.SUBREGIONS}
        for sub in Constants.SUBREGIONS:
            for y in year_sorted:
                if y in data_dict[sub]:
                    trimmed_dict[sub].append(data_dict[sub][y][1])
                else:
                    trimmed_dict[sub].append(0.0)

        if len(year_sorted) < sequence_length:
            print(f"Data tidak cukup (punya {len(year_sorted)}, butuh {sequence_length}).")
            return None

        history_years = year_sorted[-sequence_length:]
        last_year = history_years[-1]
        trimmed_final = {sub: vals[-sequence_length:] for sub, vals in trimmed_dict.items()}
        
        return trimmed_final, history_years, last_year

    except Exception as e:
        print(f"Gagal mengambil sequence LSTM: {e}", file=sys.stderr)
        return None
    finally:
        if conn:
            conn.close()
#endregion


#region Save Model
def save_model_to_db(model_object, model_type: str, metrics: dict):
    conn = None
    try:
        print("\nMenghubungkan dengan database...\n")
        conn = psycopg2.connect(**Constants.DB_CONFIG)
        cursor = conn.cursor()

        model_binary = None
        # Serialisasi model berdasarkan tipenya
        if model_type in ["LogReg", "LSTM"]:
            mem_file = io.BytesIO()
            if model_type == "LSTM":
                keras_model = model_object['model']
                with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                    keras_model.save(tmp_path)
                    with open(tmp_path, "rb") as f:
                        model_bytes = f.read()
                    model_object['model'] = model_bytes

            dump(model_object, mem_file)
            mem_file.seek(0)
            model_binary = mem_file.read()
        else:
            raise ValueError(f"Tipe model tidak didukung: {model_type}")
        
        if model_type == "LSTM" and 'R2GraphPath' in metrics and metrics['R2GraphPath']:
            graph_temp_path = metrics['R2GraphPath']
            if os.path.exists(graph_temp_path):
                try:
                    with open(graph_temp_path, "rb") as f_graph:
                        graph_bytes = f_graph.read()
                    graph_base64 = base64.b64encode(graph_bytes).decode('utf-8')
                    metrics['R2GraphPath'] = graph_base64 
                    os.remove(graph_temp_path)
                except Exception as e:
                    metrics['R2GraphPath'] = None
            else:
                metrics['R2GraphPath'] = None

        metrics_json = json.dumps(metrics)
        train_date = datetime.now()
        
        timestamp_str = train_date.strftime("%Y%m%d_%H%M%S")
        model_name = f"{model_type.lower()}_model_{timestamp_str}"

        insert_query = """
            INSERT INTO model (model_name, model_type, train_date, model_obj, metrics)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id_model;
        """
        
        cursor.execute(insert_query, (model_name, model_type, train_date, psycopg2.Binary(model_binary), metrics_json))
        model_id = cursor.fetchone()[0]

        conn.commit()
        print(f"Model '{model_name}' berhasil disimpan ke database dengan ID: {model_id}")
        return model_id

    except psycopg2.Error as e:
        print(f"Error from database: {e}", file=sys.stderr)
        if conn:
            conn.rollback()
        return None
    except Exception as e:
        print(f"Exception error: {e}", file=sys.stderr)
        return None
    finally:
        if conn:
            conn.close()
#endregion


#region Load Model
def load_model_from_db(model_type: str):
    conn = None
    try:
        print("\nMenghubungkan dengan database...\n")
        conn = psycopg2.connect(**Constants.DB_CONFIG)
        cursor = conn.cursor()

        query = """
            SELECT model_obj, id_model 
            FROM model 
            WHERE model_type = %s
            ORDER BY train_date DESC
            LIMIT 1;
        """
        cursor.execute(query, (model_type,))
        record = cursor.fetchone()

        if record:
            model_binary, latest_model_id = record[0], record[1]
            mem_file = io.BytesIO(model_binary)
            model_object = None
            
            print(f"Memuat model tipe '{model_type}' dengan ID {latest_model_id}...")
            
            if model_type in ["LogReg", "LSTM"]:
                model_object = load(mem_file)
                if model_type == "LSTM":
                    model_bytes = model_object['model']
                    import tempfile, os
                    from tensorflow import keras

                    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp_file:
                        tmp_file.write(model_bytes)
                        tmp_file.flush()
                        tmp_path = tmp_file.name

                    # try:
                        model_object['model'] = keras.models.load_model(tmp_path)
                    # finally:
                    #     if os.path.exists(tmp_path):
                    #         os.remove(tmp_path)
            else:
                print(f"Tipe model tidak dikenal: '{model_type}'. Tidak dapat memuat model.", file=sys.stderr)
                return None
            
            print("Model berhasil dimuat dari database.")
            return model_object, latest_model_id
        else:
            print(f"Model dengan ID {model_type} tidak ditemukan di database.")
            return None

    except psycopg2.Error as e:
        print(f"Database error saat memuat model: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Exception error: {e}", file=sys.stderr)
        return None
    finally:
        if conn:
            conn.close()
#endregion