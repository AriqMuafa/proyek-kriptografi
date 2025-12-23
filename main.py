import json
import os
import io
import base64
import time
from itertools import cycle
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# --- LOGIKA KRIPTOGRAFI INTI ---
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

# --- DATA MATRIKS (CONTOH) ---
# ⚠️ PENTING: Salin semua matriks dari notebook asli Anda ke sini.
# Pastikan KEY dictionary sesuai dengan ID yang ada di file JSON (misal "K44", "AES_STD")
PAPER_MATRICES = {
    "AES_STD": np.array([
        [1, 0, 0, 0, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 1, 1, 1],
        [1, 1, 1, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 1, 1]
    ], dtype=int),
    "K44": bin_to_matrix([
        "01010111", "10101011", "11010101", "11101010",
        "01110101", "10111010", "01011101", "10101110"
    ]),
     "K1": bin_to_matrix([
        "00000001", "10000000", "01000000", "00100000",
        "00010000", "00001000", "00000100", "00000010"
    ]),
    # ... TAMBAHKAN K2 - K128 DARI PDF ANDA DI SINI ...
}

def get_sbox_by_id(sbox_id):
    # Mapping ID dari JSON ke Key PAPER_MATRICES jika perlu
    # Disini kita asumsikan ID di JSON sama dengan key di PAPER_MATRICES
    # Jika tidak ada, fallback ke AES
    matrix = PAPER_MATRICES.get(sbox_id, PAPER_MATRICES["AES_STD"])
    return create_sbox(matrix, AES_CONSTANT)

# --- ENDPOINTS ---

@app.get("/", response_class=HTMLResponse)
def home():
    index_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Error: index.html not found</h1>"

@app.get("/api/data")
def get_data():
    return SBOX_DATA

@app.get("/api/sbox-values/{sbox_id}")
def get_sbox_values(sbox_id: str):
    """Mengembalikan nilai S-Box array (0-255) untuk visualisasi grid"""
    sbox = get_sbox_by_id(sbox_id)
    return {"sbox": sbox}

# --- TEXT ENCRYPTION/DECRYPTION ---

class TextCryptoRequest(BaseModel):
    text: str
    key: str
    sbox_id: str

@app.post("/api/encrypt-text")
def encrypt_text_api(payload: TextCryptoRequest):
    try:
        sbox = get_sbox_by_id(payload.sbox_id)
        key_bytes = payload.key.encode('utf-8')
        if not key_bytes:
            return JSONResponse({"error": "Key required"}, status_code=400)
        
        # Logic: SBox[Char ^ Key]
        encrypted_bytes = []
        for i, char in enumerate(payload.text):
            char_code = ord(char)
            # Batasi input ke ASCII standard untuk demo ini, atau biarkan utf-8
            # Simple XOR mixing with key loop
            k = key_bytes[i % len(key_bytes)]
            val = sbox[char_code ^ k] if char_code < 256 else char_code
            encrypted_bytes.append(val)
        
        # Return as Hex String
        hex_output = " ".join([f"{b:02X}" for b in encrypted_bytes])
        return {"result": hex_output}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/decrypt-text")
def decrypt_text_api(payload: TextCryptoRequest):
    try:
        sbox = get_sbox_by_id(payload.sbox_id)
        # Generate Inverse SBox
        inv_sbox = [0] * 256
        for i, val in enumerate(sbox):
            inv_sbox[val] = i
            
        key_bytes = payload.key.encode('utf-8')
        if not key_bytes:
            return JSONResponse({"error": "Key required"}, status_code=400)

        # Parse Hex String
        try:
            hex_values = [int(x, 16) for x in payload.text.strip().split()]
        except:
             return JSONResponse({"error": "Invalid Hex Format"}, status_code=400)

        decrypted_chars = []
        for i, val in enumerate(hex_values):
            if val > 255: continue
            k = key_bytes[i % len(key_bytes)]
            # Inverse Logic: InvSBox[Val] ^ Key
            original_val = inv_sbox[val] ^ k
            decrypted_chars.append(chr(original_val))
            
        return {"result": "".join(decrypted_chars)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# --- IMAGE PROCESSING ---

@app.post("/api/process-image")
async def process_image(
    file: UploadFile = File(...), 
    sbox_id: str = Form(...),
    mode: str = Form(...) # 'encrypt' or 'decrypt'
):
    try:
        contents = await file.read()
        sbox = get_sbox_by_id(sbox_id)
        
        # Setup Mapping Array
        if mode == 'decrypt':
            inv_sbox = [0] * 256
            for i, val in enumerate(sbox):
                inv_sbox[val] = i
            mapping = np.array(inv_sbox, dtype=np.uint8)
        else:
            mapping = np.array(sbox, dtype=np.uint8)

        # Process Image
        image = Image.open(io.BytesIO(contents))
        img_array = np.array(image)
        
        # Handle channels
        if img_array.ndim == 2: # Grayscale
             processed_array = mapping[img_array]
        else: # RGB/RGBA
             processed_array = mapping[img_array]

        result_image = Image.fromarray(processed_array)
        
        # Return Base64
        buffered = io.BytesIO()
        result_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return JSONResponse({
            "status": "success",
            "image_data": f"data:image/png;base64,{img_str}"
        })
        
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
