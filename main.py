import json
import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

# Mendapatkan path absolut direktori saat ini
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. PASTIKAN NAMA FILE SESUAI (Gunakan 'x' jika di GitHub namanya xbox)
JSON_PATH = os.path.join(BASE_DIR, 'sbox_data_full.json') 

# 2. LOAD DATA SEKALI SAJA SAAT STARTUP (Agar Cepat)
try:
    with open(JSON_PATH, 'r') as f:
        SBOX_DATA = json.load(f)
    print("✅ Data S-Box berhasil dimuat.")
except Exception as e:
    SBOX_DATA = {"error": f"Gagal memuat file JSON: {str(e)}"}
    print(f"❌ Error: {str(e)}")

@app.get("/api/data")
def get_data():
    """Mengirimkan data JSON ke frontend"""
    return SBOX_DATA

@app.get("/", response_class=HTMLResponse)
def home():
    """Menampilkan halaman dashboard utama"""
    index_path = os.path.join(BASE_DIR, "index.html")
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"<html><body><h1>Error: index.html tidak ditemukan</h1><p>{str(e)}</p></body></html>"
