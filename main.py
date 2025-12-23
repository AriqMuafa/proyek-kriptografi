import json
import os
import io
import base64
import time
from itertools import cycle
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Enable CORS - PENTING untuk production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- KONFIGURASI PATH ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE_DIR, 'sbox_analysis_complete.json')
HTML_PATH = os.path.join(BASE_DIR, 'index.html')

# --- LOAD DATA JSON ---
try:
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        SBOX_DATA = json.load(f)
    print(f"✅ Data S-Box berhasil dimuat: {len(SBOX_DATA.get('candidates', []))} candidates")
except Exception as e:
    print(f"❌ Error loading JSON: {e}")
    SBOX_DATA = {
        "metadata": {
            "generated_at": "2025-12-23",
            "total_sboxes": 0,
            "description": "Error loading data"
        },
        "candidates": []
    }

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

# --- DATA MATRIKS ---
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
    "K1": bin_to_matrix([
        "00000001", "10000000", "01000000", "00100000",
        "00010000", "00001000", "00000100", "00000010"
    ]),
    "K2": bin_to_matrix([
        "00000010", "00000001", "10000000", "01000000",
        "00100000", "00010000", "00001000", "00000100"
    ]),
    "K3": bin_to_matrix([
        "00000100", "00000010", "00000001", "10000000",
        "01000000", "00100000", "00010000", "00001000"
    ]),
    "K5": bin_to_matrix([
        "00001000", "00000100", "00000010", "00000001",
        "10000000", "01000000", "00100000", "00010000"
    ]),
    "K6": bin_to_matrix([
        "00001011", "10000101", "11000010", "01100001",
        "10110000", "01011000", "00101100", "00010110"
    ]),
    "K7": bin_to_matrix([
        "00001101", "10000110", "01000011", "10100001",
        "11010000", "01101000", "00110100", "00011010"
    ]),
    "K9": bin_to_matrix([
        "00010000", "00001000", "00000100", "00000010",
        "00000001", "10000000", "01000000", "00100000"
    ]),
    "K10": bin_to_matrix([
        "00010011", "10001001", "11000100", "01100010",
        "00110001", "10011000", "01001100", "00100110"
    ]),
    "K11": bin_to_matrix([
        "00010101", "10001010", "01000101", "10100010",
        "01010001", "10101000", "01010100", "00101010"
    ]),
    "K12": bin_to_matrix([
        "00011001", "10001100", "01000110", "00100011",
        "10010001", "11001000", "01100100", "00110010"
    ]),
    "K13": bin_to_matrix([
        "10001111", "11000111", "11100011", "11110001",
        "11111000", "01111100", "00111110", "00011111"
    ]),
    "K14": bin_to_matrix([
        "00011010", "00001101", "10000110", "01000011",
        "10100001", "11010000", "01101000", "00110100"
    ]),
    "K15": bin_to_matrix([
        "00011100", "00001110", "00000111", "10000011",
        "11000001", "11100000", "01110000", "00111000"
    ]),
    "K16": bin_to_matrix([
        "00011111", "10001111", "11000111", "11100011",
        "11110001", "11111000", "01111100", "00111110"
    ]),
    "K17": bin_to_matrix([
        "00100000", "00010000", "00001000", "00000100",
        "00000010", "00000001", "10000000", "01000000"
    ]),
    "K19": bin_to_matrix([
        "00100101", "10010010", "01001001", "10100100",
        "01010010", "00101001", "10010100", "01001010"
    ]),
    "K20": bin_to_matrix([
        "00100110", "00010011", "10001001", "11000100",
        "01100010", "00110001", "10011000", "01001100"
    ]),
    "K21": bin_to_matrix([
        "00101001", "10010100", "01001010", "00100101",
        "10010010", "01001001", "10100100", "01010010"
    ]),
    "K22": bin_to_matrix([
        "00101010", "00010101", "10001010", "01000101",
        "10100010", "01010001", "10101000", "01010100"
    ]),
    "K44": bin_to_matrix([
        "01010111", "10101011", "11010101", "11101010",
        "01110101", "10111010", "01011101", "10101110"
    ]),
    "K81": bin_to_matrix([
        "10100001", "11010000", "01101000", "00110100",
        "00011010", "00001101", "10000110", "01000011"
    ]),
    "K111": bin_to_matrix([
        "11011100", "01101110", "00110111", "10011011",
        "11001101", "11100110", "01110011", "10111001"
    ]),
    "K127": bin_to_matrix([
        "11111101", "11111110", "01111111", "10111111",
        "11011111", "11101111", "11110111", "11111011"
    ]),
    "K128": bin_to_matrix([
        "11111110", "01111111", "10111111", "11011111",
        "11101111", "11110111", "11111011", "11111101"
    ])
}

# Tambahkan mapping untuk K_Rand* dari JSON
print(f"📊 Total matrices loaded: {len(PAPER_MATRICES)}")

def get_sbox_by_id(sbox_id):
    """Get S-box by ID, generate if needed"""
    # Coba ambil dari PAPER_MATRICES
    if sbox_id in PAPER_MATRICES:
        matrix = PAPER_MATRICES[sbox_id]
        return create_sbox(matrix, AES_CONSTANT)
    
    # Jika K_Rand*, generate random (atau fallback ke AES)
    if sbox_id.startswith('K_Rand'):
        # Untuk random, kita gunakan seed dari ID untuk konsistensi
        try:
            rand_num = int(sbox_id.replace('K_Rand', ''))
            np.random.seed(rand_num + 42)  # Seed konsisten
            matrix = np.random.randint(0, 2, (8, 8))
            return create_sbox(matrix, AES_CONSTANT)
        except:
            pass
    
    # Fallback ke AES Standard
    print(f"⚠️ S-box ID '{sbox_id}' not found, using AES_STD")
    return create_sbox(PAPER_MATRICES["AES_STD"], AES_CONSTANT)

# --- ENDPOINTS ---

@app.get("/")
async def home():
    """Serve index.html"""
    if os.path.exists(HTML_PATH):
        return FileResponse(HTML_PATH, media_type='text/html')
    return HTMLResponse("""
        <html>
            <body>
                <h1>S-Box Analyzer API is Running! ✅</h1>
                <p>⚠️ index.html not found. Please add index.html file.</p>
                <p>Available endpoints:</p>
                <ul>
                    <li>GET /api/data - Get all S-box data</li>
                    <li>GET /api/sbox-values/{sbox_id} - Get S-box values</li>
                    <li>POST /api/encrypt-text - Encrypt text</li>
                    <li>POST /api/decrypt-text - Decrypt text</li>
                    <li>POST /api/process-image - Process image</li>
                </ul>
            </body>
        </html>
    """)

@app.get("/api/data")
async def get_data():
    """Return S-box data as JSON"""
    try:
        return JSONResponse(content=SBOX_DATA)
    except Exception as e:
        print(f"Error in /api/data: {e}")
        return JSONResponse(
            content={"error": str(e), "candidates": []},
            status_code=500
        )

@app.get("/api/sbox-values/{sbox_id}")
async def get_sbox_values(sbox_id: str):
    """Return S-box array values"""
    try:
        sbox = get_sbox_by_id(sbox_id)
        return JSONResponse(content={"sbox": sbox})
    except Exception as e:
        print(f"Error in /api/sbox-values: {e}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

# --- TEXT ENCRYPTION/DECRYPTION ---

class TextCryptoRequest(BaseModel):
    text: str
    key: str
    sbox_id: str

@app.post("/api/encrypt-text")
async def encrypt_text_api(payload: TextCryptoRequest):
    try:
        sbox = get_sbox_by_id(payload.sbox_id)
        key_bytes = payload.key.encode('utf-8')
        
        if not key_bytes:
            return JSONResponse({"error": "Key required"}, status_code=400)
        
        encrypted_bytes = []
        for i, char in enumerate(payload.text):
            char_code = ord(char)
            k = key_bytes[i % len(key_bytes)]
            
            # XOR with key, then S-box substitution
            xor_val = char_code ^ k
            if xor_val < 256:
                encrypted_bytes.append(sbox[xor_val])
            else:
                encrypted_bytes.append(char_code)
        
        hex_output = " ".join([f"{b:02X}" for b in encrypted_bytes])
        return JSONResponse({"result": hex_output})
        
    except Exception as e:
        print(f"Error in encrypt-text: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/decrypt-text")
async def decrypt_text_api(payload: TextCryptoRequest):
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
            if val > 255: 
                continue
            k = key_bytes[i % len(key_bytes)]
            
            # Inverse: InvSBox[val] then XOR with key
            original_val = inv_sbox[val] ^ k
            decrypted_chars.append(chr(original_val))
            
        return JSONResponse({"result": "".join(decrypted_chars)})
        
    except Exception as e:
        print(f"Error in decrypt-text: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

# --- IMAGE PROCESSING ---

@app.post("/api/process-image")
async def process_image(
    file: UploadFile = File(...), 
    sbox_id: str = Form(...),
    mode: str = Form(...)
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
        if img_array.ndim == 2:  # Grayscale
            processed_array = mapping[img_array]
        else:  # RGB/RGBA
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
        print(f"Error in process-image: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

# Health check endpoint
@app.get("/health")
async def health_check():
    return JSONResponse({
        "status": "healthy",
        "sbox_count": len(SBOX_DATA.get('candidates', [])),
        "matrices_loaded": len(PAPER_MATRICES)
    })

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
