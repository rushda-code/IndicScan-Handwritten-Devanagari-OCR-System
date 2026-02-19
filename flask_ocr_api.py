# -*- coding: utf-8 -*-
"""
Created on Tue Dec 30 19:10:02 2025

@author: Admin
"""
#flask_ocr_api.py
# =====================================================
# Flask OCR API (FINAL, NO LLM, UI-SAFE)
# Uses ocr_router.py for all OCR logic
# =====================================================

from flask import Flask, request, jsonify
import os
import tempfile
from ocr_router import run_ocr

app = Flask(__name__)

# ===================== HEALTH CHECK =====================
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "OCR backend running"}), 200

# ===================== OCR ENDPOINT =====================
@app.route("/ocr", methods=["POST"])
def ocr():
    """
    Expects:
      - image (file)
      - mode: 'char' or 'word'
    """

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    mode = request.form.get("mode", "").lower()
    if mode not in ["char", "word"]:
        return jsonify({"error": "Invalid mode (use 'char' or 'word')"}), 400

    image_file = request.files["image"]

    # Save image temporarily (ASCII-safe)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image_path = tmp.name
        image_file.save(image_path)

    try:
        text = run_ocr(mode, image_path)
    except Exception as e:
        os.remove(image_path)
        return jsonify({"error": str(e)}), 500

    os.remove(image_path)

    return jsonify({
        "mode": mode,
        "ocr_output": text
    })

# ===================== RUN =====================
if __name__ == "__main__":
    print("🚀 Flask OCR backend running at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
