import json
import os
import io
import base64
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE_DIR, 'sbox_data_full.json')

# --- BAGIAN 1: KONFIGURASI DAN DATA ---
# Load Data JSON S-Box (Metadata)
try:
    with open(JSON_PATH, 'r') as f:
        SBOX_DATA = json.load(f)
    print("✅ Data S-Box berhasil dimuat.")
except Exception as e:
    SBOX_DATA = {"error": f"Gagal memuat file JSON: {str(e)}"}

# PENTING: Anda harus menyalin dictionary PAPER_MATRICES dari notebook asli ke sini.
# Agar server bisa membuat S-Box berdasarkan pilihan user (K_1, K_44, dll).
# Saya sertakan contoh kecil, SILAKAN COPY LENGKAP DARI NOTEBOOK ANDA.
def bin_to_matrix(bin_list):
    matrix = []
    for row_str in bin_list:
        row = [int(b) for b in row_str]
        matrix.append(row)
    return np.array(matrix, dtype=int)

# [COPY DARI PDF HALAMAN 3-6]
PAPER_MATRICES = {
    "K_44": bin_to_matrix([
        "01010111", "10101011", "01110101", "10111010",
        "10001011", "01000101", "10101000", "01010100"
    ]),
    # ... TAMBAHKAN K_1 SAMPAI K_128 DARI KODE ASLI ANDA DI SINI ...
}

AES_CONSTANT = 0x63

# --- BAGIAN 2: FUNGSI MATEMATIKA INTI (Dari PDF Hal 1-2) ---
def gf_multiply(a, b):
    p = 0
    for i in range(8):
        if b & 1: p ^= a
        hi_bit_set = a & 0x80
        a <<= 1
        if hi_bit_set: a ^= 0x11B # Irreducible polynomial
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

# --- BAGIAN 3: API ENDPOINTS ---

@app.get("/api/data")
def get_data():
    return SBOX_DATA

@app.post("/api/encrypt-image")
async def encrypt_image_endpoint(
    file: UploadFile = File(...), 
    sbox_name: str = Form(...)
):
    """Menerima upload gambar, mengenkripsi dengan S-box pilihan, mengembalikan Base64"""
    try:
        # 1. Validasi Input
        if sbox_name not in PAPER_MATRICES:
            # Jika user memilih random/lainnya, kita bisa pakai default atau error
            # Untuk demo, kita fallback ke K_44 jika tidak ada
            matrix = PAPER_MATRICES.get("K_44") 
        else:
            matrix = PAPER_MATRICES[sbox_name]

        # 2. Generate S-box on the fly
        sbox = create_sbox(matrix, AES_CONSTANT)

        # 3. Baca Gambar
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        img_array = np.array(image)
        original_shape = img_array.shape

        # 4. Proses Enkripsi (Logic dari PDF Hal 21)
        # Handle Grayscale vs RGB
        if len(original_shape) == 2:
            flat_img = img_array.flatten()
        else:
            flat_img = img_array.reshape(-1)
            
        # Apply Substitution
        # Pastikan tipe data uint8 agar valid sebagai image
        encrypted_flat = np.array([sbox[pixel] for pixel in flat_img], dtype=np.uint8)
        
        # Kembalikan ke bentuk asal
        encrypted_array = encrypted_flat.reshape(original_shape)
        encrypted_image = Image.fromarray(encrypted_array)

        # 5. Konversi ke Base64 untuk dikirim ke Frontend
        buffered = io.BytesIO()
        # Simpan sebagai PNG agar tidak ada kompresi lossy yang merusak enkripsi
        encrypted_image.save(buffered, format="PNG") 
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return JSONResponse({
            "status": "success",
            "sbox_used": sbox_name,
            "image_base64": f"data:image/png;base64,{img_str}"
        })

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.get("/", response_class=HTMLResponse)
def home():
    index_path = os.path.join(BASE_DIR, "index.html")
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error loading index.html: {str(e)}"
