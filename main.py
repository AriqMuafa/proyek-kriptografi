import json
import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI()

# Membaca data JSON
def load_data():
    with open('sbox_data_full.json', 'r') as f:
        return json.load(f)

@app.get("/api/data")
def get_data():
    return load_data()

@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html", "r") as f:
        return f.read()
