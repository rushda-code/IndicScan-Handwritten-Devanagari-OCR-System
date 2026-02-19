# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 18:48:08 2025

@author: Admin
"""

import os
import time
import cv2
import pickle
import numpy as np
import tensorflow as tf
import sys
from datetime import datetime
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint

# ================= PATHS =================
BASE_DIR = r"D:/SEM III/1. Min Pro/DATASETS"
WORD_FINAL = BASE_DIR + "/word_final"
OUT_DIR = BASE_DIR + "/MAIN/output"
LOG_FILE = OUT_DIR + "/train_word_crnn.log"

os.makedirs(OUT_DIR, exist_ok=True)

# ================= LOGGING =================
class Logger:
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

sys.stdout = Logger(LOG_FILE)

print("🕒 Training started at:", datetime.now())
print("📁 Output dir:", OUT_DIR)

# ================= CONSTANTS =================
IMG_H = 128
IMG_W = 384
BATCH_SIZE = 8
EPOCHS = 25

# ================= LOAD DATA =================
print("\n📦 Loading encoded data...")

X_paths = np.load(WORD_FINAL + "/X_paths.npy", allow_pickle=True)
y_seq   = np.load(WORD_FINAL + "/y_words_seq.npy", allow_pickle=True)
y_len   = np.load(WORD_FINAL + "/y_words_len.npy", allow_pickle=True)

with open(WORD_FINAL + "/idx2char.pkl", "rb") as f:
    idx2char = pickle.load(f)

VOCAB_SIZE = len(idx2char)

print(f"🖼 Total samples : {len(X_paths)}")
print(f"🔤 Total classes : {VOCAB_SIZE}")
print(f"📏 Avg label len : {int(sum(y_len) / len(y_len))}")

# ================= IMAGE LOADER =================
def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("❌ Failed to load image:", path)
        return None

    h, w = img.shape
    scale = IMG_H / h
    new_w = min(int(w * scale), IMG_W)

    img = cv2.resize(img, (new_w, IMG_H))

    canvas = np.ones((IMG_H, IMG_W), dtype=np.uint8) * 255
    canvas[:, :new_w] = img

    img = canvas.astype("float32") / 255.0
    return img[..., None]

# ================= DATA GENERATOR =================
def data_generator():
    while True:
        for i in range(0, len(X_paths), BATCH_SIZE):
            batch_x, batch_y, batch_il, batch_ll = [], [], [], []

            for j in range(i, min(i + BATCH_SIZE, len(X_paths))):
                img = load_image(X_paths[j])
                if img is None:
                    continue

                batch_x.append(img)
                batch_y.append(y_seq[j])
                batch_il.append([IMG_W // 4])
                batch_ll.append([y_len[j]])

            if len(batch_x) == 0:
                continue

            yield (
                {
                    "image": np.array(batch_x),
                    "label": tf.keras.preprocessing.sequence.pad_sequences(
                        batch_y, padding="post"
                    ),
                    "input_length": np.array(batch_il),
                    "label_length": np.array(batch_ll),
                },
                np.zeros(len(batch_x))  # dummy y
            )

# ================= CTC LOSS =================
def ctc_lambda(args):
    y_true, y_pred, input_len, label_len = args
    return K.ctc_batch_cost(y_true, y_pred, input_len, label_len)

# ================= MODEL =================
print("\n🧠 Building CRNN model...")

image = layers.Input((IMG_H, IMG_W, 1), name="image")
label = layers.Input((None,), dtype="int32", name="label")
input_len = layers.Input((1,), dtype="int32", name="input_length")
label_len = layers.Input((1,), dtype="int32", name="label_length")

x = layers.Conv2D(64, 3, padding="same", activation="relu")(image)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Reshape((IMG_W // 4, (IMG_H // 4) * 128))(x)
x = layers.Dense(128, activation="relu")(x)

x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)

y_pred = layers.Dense(VOCAB_SIZE, activation="softmax", name="softmax")(x)

ctc_out = layers.Lambda(ctc_lambda, name="ctc")(
    [label, y_pred, input_len, label_len]
)

train_model = models.Model(
    inputs=[image, label, input_len, label_len],
    outputs=ctc_out
)

# 🔴 CTC COMPILE FIX 🔴
train_model.compile(
    optimizer="adam",
    loss=lambda y_true, y_pred: y_pred
)

train_model.summary()

# ================= CALLBACKS =================
csv_logger = CSVLogger(OUT_DIR + "/training_log.csv", append=True)

ckpt = ModelCheckpoint(
    OUT_DIR + "/best_train_model.keras",
    monitor="loss",
    save_best_only=True,
    save_weights_only=False
)

# ================= TRAIN =================
print("\n🚀 Training started (CPU-safe)")
start = time.time()

train_model.fit(
    data_generator(),
    steps_per_epoch=len(X_paths) // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[csv_logger, ckpt],
    verbose=2   # 👈 KEY FIX
)


print(f"\n✅ Training finished in {(time.time() - start)/60:.2f} minutes")

# ================= SAVE INFERENCE MODEL =================
print("\n💾 Saving inference-only model...")

infer_model = models.Model(inputs=image, outputs=y_pred)
infer_model.save(OUT_DIR + "/word_crnn_infer.keras")

print("✅ Inference model saved at:")
print(OUT_DIR + "/word_crnn_infer.keras")
print("🕒 Training ended at:", datetime.now())
print("📁 Training log saved to:", LOG_FILE)
