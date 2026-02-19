# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 23:51:41 2025

@author: Admin
"""
#sample_char_dataset.py
import os
import pickle
import numpy as np
from collections import defaultdict
from datetime import datetime
import random

# ================= PATHS =================
BASE_DIR = r"D:/SEM III/1. Min Pro/DATASETS"
CHAR_FINAL = BASE_DIR + "/char_final"

X_FULL = CHAR_FINAL + "/X_char_paths.npy"
Y_FULL = CHAR_FINAL + "/y_char_seq.npy"
LEN_FULL = CHAR_FINAL + "/y_char_len.npy"

X_OUT = CHAR_FINAL + "/X_char_paths_sampled.npy"
Y_OUT = CHAR_FINAL + "/y_char_seq_sampled.npy"
LEN_OUT = CHAR_FINAL + "/y_char_len_sampled.npy"

LOG_PATH = CHAR_FINAL + "/sample_char_dataset.log"

SAMPLES_PER_CHAR = 1000

# ================= LOAD =================
print("📦 Loading full encoded dataset...")
X_paths = np.load(X_FULL, allow_pickle=True)
y_seq = np.load(Y_FULL, allow_pickle=True)
y_len = np.load(LEN_FULL, allow_pickle=True)

print(f"🖼 Total samples loaded: {len(X_paths)}")

# ================= GROUP BY CHARACTER =================
char_buckets = defaultdict(list)

for idx in range(len(X_paths)):
    char_id = y_seq[idx][0]   # single char
    char_buckets[char_id].append(idx)

# ================= SAMPLE =================
sampled_indices = []

for char_id, indices in char_buckets.items():
    if len(indices) >= SAMPLES_PER_CHAR:
        sampled = random.sample(indices, SAMPLES_PER_CHAR)
    else:
        sampled = indices  # take all if less than 1000

    sampled_indices.extend(sampled)

# Shuffle final set
random.shuffle(sampled_indices)

# ================= BUILD SAMPLED ARRAYS =================
X_sampled = X_paths[sampled_indices]
y_sampled = y_seq[sampled_indices]
len_sampled = y_len[sampled_indices]

# ================= SAVE =================
np.save(X_OUT, X_sampled)
np.save(Y_OUT, y_sampled)
np.save(LEN_OUT, len_sampled)

# ================= LOG =================
with open(LOG_PATH, "w", encoding="utf-8") as log:
    log.write("CHAR DATASET SAMPLING LOG\n")
    log.write("="*60 + "\n")
    log.write(f"Start time           : {datetime.now()}\n")
    log.write(f"Samples per char     : {SAMPLES_PER_CHAR}\n")
    log.write(f"Original samples     : {len(X_paths)}\n")
    log.write(f"Sampled samples      : {len(X_sampled)}\n\n")

    for char_id, indices in char_buckets.items():
        log.write(
            f"Char ID {char_id:02d} : {min(len(indices), SAMPLES_PER_CHAR)}\n"
        )

# ================= PRINT =================
print("✅ Sampling complete")
print(f"🎯 Samples per char : {SAMPLES_PER_CHAR}")
print(f"🧮 Total samples    : {len(X_sampled)}")
print(f"📁 Saved:")
print(f"   - {X_OUT}")
print(f"   - {Y_OUT}")
print(f"   - {LEN_OUT}")
print(f"📄 Log file         : {LOG_PATH}")
