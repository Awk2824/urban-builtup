# Target resolusi setelah resampling (dalam meter)
TARGET_RESOLUTION = 10

# Sub-wilayah
SUBREGIONS = ["NW", "NE", "SE", "SW"]

# Faktor skala reflektansi Sentinel-2
SCALE_FACTOR = 0.0001

# Threshold ENDISI
ENDISI_THRESHOLD = -0.15

# Konstanta
EPSILON = 1e-6
NORMALIZE_FEATURES = True

# .env Config
DB_CONFIG = {
    'dbname': 'build_up_land',
    'user': 'postgres',
    'password': '123456',
    'host': 'localhost',
    'port': '5432'
}