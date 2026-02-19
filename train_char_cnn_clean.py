# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 23:12:25 2025

@author: Admin
"""
#train_char_cnn_clean.py
import os
import time
import cv2
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from datetime import datetime

# ================= PATHS =================
BASE_DIR = r"D:/SEM III/1. Min Pro/DATASETS"
CHAR_FINAL = BASE_DIR + "/char_final"
OUT_DIR = CHAR_FINAL + "/models"

os.makedirs(OUT_DIR, exist_ok=True)

X_PATHS = CHAR_FINAL + "/X_char_paths_sampled.npy"
Y_SEQ   = CHAR_FINAL + "/y_char_seq_sampled.npy"
CHAR2IDX_PATH = CHAR_FINAL + "/char2idx.pkl"

# ================= CONSTANTS =================
IMG_H = 64
IMG_W = 64
BATCH_SIZE = 64
EPOCHS = 10   # SAFE on CPU

# ================= LOAD DATA =================
print("📦 Loading sampled data...")

X_paths = np.load(X_PATHS, allow_pickle=True)
y_seq   = np.load(Y_SEQ, allow_pickle=True)

with open(CHAR2IDX_PATH, "rb") as f:
    char2idx = pickle.load(f)

NUM_CLASSES = len(char2idx)

print(f"🖼 Samples : {len(X_paths)}")
print(f"🔤 Classes : {NUM_CLASSES}")

# ================= IMAGE LOADER =================
def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_W, IMG_H))
    img = img.astype("float32") / 255.0
    return img[..., None]

# ================= DATA GENERATOR =================
def data_generator():
    while True:
        idxs = np.random.permutation(len(X_paths))
        for i in range(0, len(idxs), BATCH_SIZE):
            batch_idx = idxs[i:i+BATCH_SIZE]
            batch_x, batch_y = [], []

            for j in batch_idx:
                img = load_image(X_paths[j])
                if img is None:
                    continue
                batch_x.append(img)
                batch_y.append(y_seq[j][0])  # single char

            if not batch_x:
                continue

            yield np.array(batch_x), tf.keras.utils.to_categorical(
                batch_y, NUM_CLASSES
            )

# ================= MODEL =================
model = models.Sequential([
    layers.Input((IMG_H, IMG_W, 1)),

    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(2),

    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(2),

    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(2),

    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.4),

    layers.Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ================= CALLBACKS =================
run_id = "char_cnn_sampled_" + datetime.now().strftime("%Y%m%d_%H%M%S")

csv_logger = CSVLogger(
    os.path.join(OUT_DIR, f"{run_id}_training_log.csv")
)

ckpt_each_epoch = ModelCheckpoint(
    filepath=os.path.join(OUT_DIR, "epoch_{epoch:02d}.keras"),
    save_freq="epoch"
)

ckpt_best = ModelCheckpoint(
    filepath=os.path.join(OUT_DIR, "best_model.keras"),
    monitor="accuracy",
    save_best_only=True
)

# ================= TRAIN =================
print("\n🚀 Training started (sampled, CPU-safe)")
start = time.time()

model.fit(
    data_generator(),
    steps_per_epoch=len(X_paths) // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[csv_logger, ckpt_each_epoch, ckpt_best]
)

print(f"\n✅ Training finished in {(time.time() - start)/60:.2f} minutes")

# ================= FINAL SAVE =================
final_path = os.path.join(OUT_DIR, "final_model.keras")
model.save(final_path)

print("\n💾 Models saved:")
print(f"   - {OUT_DIR}/epoch_XX.keras")
print(f"   - {OUT_DIR}/best_model.keras")
print(f"   - {final_path}")
