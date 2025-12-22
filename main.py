import json
import os
import io
import base64
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI()

# --- KONFIGURASI PATH ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE_DIR, 'sbox_analysis_complete.json')

# --- LOAD DATA JSON ---
try:
    with open(JSON_PATH, 'r') as f:
        SBOX_DATA = json.load(f)
    print("✅ Data S-Box berhasil dimuat.")
except Exception as e:
    print(f"❌ Error loading JSON: {e}")
    SBOX_DATA = {"candidates": []}

# --- BAGIAN 1: LOGIKA MATEMATIKA (DARI NOTEBOOK) ---
# Salin PAPER_MATRICES dan fungsi inti dari notebook Anda
# (Saya ringkas disini, pastikan Anda copy paste LENGKAP dari notebook cell 2 & 3)

AES_CONSTANT = 0x63
IRREDUCIBLE_POLY = 0x11B

def gf_multiply(a, b):
    p = 0
    for i in range(8):
        if b & 1: p ^= a
        hi_bit_set = a & 0x80
        a <<= 1
        if hi_bit_set: a ^= IRREDUCIBLE_POLY
        b >>= 1
    return p & 0xFF

def generate_inverse_table():
    inv = [0] * 256
    for i in range(1, 256):
        for j in range(1, 256):
            if gf_multiply(i, j) == 1:
                inv[i] = j
                break
    return inv

INVERSE_TABLE = generate_inverse_table()

def byte_to_bits(byte_val):
    return np.array([(byte_val >> i) & 1 for i in range(8)], dtype=int)

def bits_to_byte(bits):
    val = 0
    for i in range(8):
        if bits[i]: val |= (1 << i)
    return val

def create_sbox(matrix, constant):
    sbox = []
    matrix_np = np.array(matrix, dtype=int)
    for x in range(256):
        inv_x = INVERSE_TABLE[x]
        inv_bits = byte_to_bits(inv_x)
        trans_bits = matrix_np.dot(inv_bits) % 2
        final_val = bits_to_byte(trans_bits) ^ constant
        sbox.append(final_val)
    return sbox

def bin_to_matrix(bin_list):
    matrix = []
    for row_str in bin_list:
        row = [int(b) for b in row_str]
        matrix.append(row)
    return np.array(matrix, dtype=int)

# ⚠️ PENTING: Paste dictionary PAPER_MATRICES lengkap dari notebook Cell 3 disini
PAPER_MATRICES = {
    "K_44": bin_to_matrix([
        "01010111", "10101011", "11010101", "11101010",
        "01110101", "10111010", "01011101", "10101110"
    ]),
    # ... Tambahkan matriks lain (K_1, dll) jika perlu ...
}

# --- BAGIAN 2: LOGIKA ENKRIPSI GAMBAR (DARI CELL TERAKHIR) ---
# Fungsi ini diadaptasi dari `encrypt_image_with_sbox` di notebook

def process_image_crypto(image_bytes, sbox_name, mode='encrypt'):
    # 1. Validasi S-Box
    if sbox_name not in PAPER_MATRICES:
        # Fallback ke K_44 jika nama tidak ditemukan
        matrix = PAPER_MATRICES["K_44"]
    else:
        matrix = PAPER_MATRICES[sbox_name]

    # 2. Buat S-Box
    sbox = create_sbox(matrix, AES_CONSTANT)
    
    # Jika decrypt, kita butuh inverse sbox
    if mode == 'decrypt':
        inv_sbox = [0] * 256
        for i in range(256):
            inv_sbox[sbox[i]] = i
        mapping_array = inv_sbox
    else:
        mapping_array = sbox

    # 3. Buka Gambar
    image = Image.open(io.BytesIO(image_bytes))
    img_array = np.array(image)
    original_shape = img_array.shape

    # 4. Flatten (Mendatarkan array)
    if len(original_shape) == 2: # Grayscale
        flat_img = img_array.flatten()
    else: # RGB/RGBA
        flat_img = img_array.reshape(-1)

    # 5. Proses Substitusi (Core Logic)
    # mapping_array adalah sbox (untuk encrypt) atau inv_sbox (untuk decrypt)
    processed_flat = np.array([mapping_array[pixel] for pixel in flat_img], dtype=np.uint8)

    # 6. Kembalikan ke bentuk gambar
    processed_array = processed_flat.reshape(original_shape)
    result_image = Image.fromarray(processed_array)

    return result_image

# --- BAGIAN 3: ENDPOINTS API ---

@app.get("/")
def home():
    index_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>Error: index.html not found</h1>")

@app.get("/api/data")
def get_sbox_data():
    return SBOX_DATA

@app.post("/api/process-image")
async def process_image(
    file: UploadFile = File(...), 
    sbox: str = Form(...),
    mode: str = Form(...) # 'encrypt' atau 'decrypt'
):
    try:
        contents = await file.read()
        
        # Jalankan logika pemrosesan
        result_img = process_image_crypto(contents, sbox, mode)
        
        # Konversi hasil ke Base64 agar bisa ditampilkan di HTML
        buffered = io.BytesIO()
        result_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return JSONResponse({
            "status": "success",
            "mode": mode,
            "sbox": sbox,
            "image_data": f"data:image/png;base64,{img_str}"
        })
        
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
