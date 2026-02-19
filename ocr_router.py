# -*- coding: utf-8 -*-
"""
Created on Tue Dec 30 18:30:59 2025

@author: Admin
"""
#ocr_router.py
# =====================================================
# OCR ROUTER – FINAL (DYNAMIC, UI-READY, NO LLM)
# Supports:
#   - Character OCR (CNN)
#   - Word OCR (CRNN + CTC Beam Search)
# Accepts user image path & mode
# =====================================================

import os
import sys
import cv2
import pickle
import numpy as np
import tensorflow as tf
from datetime import datetime

# ===================== BASE DIR =====================
BASE_DIR = r"D:/SEM III/1. Min Pro/DATASETS"

# ===================== MODEL PATHS =====================
CHAR_MODEL_PATH = BASE_DIR + "/char_final/models/best_model.keras"
CHAR_IDX2CHAR   = BASE_DIR + "/char_final/idx2char.pkl"

WORD_MODEL_PATH = BASE_DIR + "/MAIN/output/word_crnn_infer.keras"
WORD_IDX2CHAR   = BASE_DIR + "/word_final/idx2char.pkl"

# ===================== LOGGING =====================
LOG_DIR = BASE_DIR + "/MAIN/logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "ocr_router.log")

class Logger:
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

sys.stdout = Logger(LOG_FILE)

print("\n" + "="*70)
print("🕒 OCR ROUTER STARTED:", datetime.now())
print("="*70)

# ===================== LOAD MODELS =====================
print("🔄 Loading models...")

char_model = tf.keras.models.load_model(CHAR_MODEL_PATH)
word_model = tf.keras.models.load_model(WORD_MODEL_PATH, compile=False)

with open(CHAR_IDX2CHAR, "rb") as f:
    char_idx2char = pickle.load(f)

with open(WORD_IDX2CHAR, "rb") as f:
    word_idx2char = pickle.load(f)

print("✅ Models loaded")
print(f"🔤 Character classes : {len(char_idx2char)}")
print(f"🔤 Word vocab size   : {len(word_idx2char)}")

# ===================== PREPROCESS =====================
def preprocess_char(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Failed to load image")

    img = cv2.resize(img, (64, 64))
    img = img.astype("float32") / 255.0
    img = img[..., None]
    return np.expand_dims(img, 0)

def preprocess_word(img_path):
    data = np.fromfile(img_path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Failed to load image")

    IMG_H, IMG_W = 128, 384
    h, w = img.shape
    scale = IMG_H / h
    new_w = min(int(w * scale), IMG_W)

    img = cv2.resize(img, (new_w, IMG_H))
    canvas = np.ones((IMG_H, IMG_W), dtype=np.uint8) * 255
    canvas[:, :new_w] = img

    img = canvas.astype("float32") / 255.0
    img = 1.0 - img
    img = img[..., None]

    return np.expand_dims(img, 0)

# ===================== OCR FUNCTIONS =====================
def run_char_ocr(img_path):
    X = preprocess_char(img_path)
    pred = char_model.predict(X, verbose=0)
    idx = int(np.argmax(pred[0]))
    return char_idx2char[idx]

def run_word_ocr(img_path):
    X = preprocess_word(img_path)
    pred = word_model.predict(X, verbose=0)

    decoded, _ = tf.keras.backend.ctc_decode(
        pred,
        input_length=np.array([pred.shape[1]]),
        greedy=False,
        beam_width=10
    )

    seq = decoded[0].numpy()[0]
    return "".join(word_idx2char[i] for i in seq if i != -1)

# ===================== MAIN ROUTER =====================
def run_ocr(mode, image_path):
    """
    mode: 'char' or 'word'
    image_path: user provided image
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError("Image path does not exist")

    print("\n📥 Image:", image_path)
    print("🔀 Mode :", mode)

    if mode == "char":
        result = run_char_ocr(image_path)

    elif mode == "word":
        result = run_word_ocr(image_path)

    else:
        raise ValueError("Invalid mode. Use 'char' or 'word'.")

    print("📝 OCR Output:", result)
    return result
