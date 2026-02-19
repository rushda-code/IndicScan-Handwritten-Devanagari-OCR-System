# IndicScan – Handwritten Devanagari OCR

IndicScan is a CPU-based handwritten Devanagari OCR system developed as part of an MSc Mini Project.

----------

## Features
- Character-level OCR using CNN (44 classes)
- Word-level OCR using CRNN + CTC (experimental)
- Flask backend API
- Streamlit user interface
- Balanced dataset sampling

## Dataset
The dataset is available on Kaggle:
[IndicScan Dataset](PASTE_YOUR_KAGGLE_LINK_HERE)

Character dataset: ~1.6 lakh images (44 classes used for training)
Word dataset: ~800 labeled handwritten words

## How to Run
1. Install requirements:
   pip install -r requirements.txt

2. Start backend:
   python flask_ocr_api.py

3. Start UI:
   streamlit run streamlit_ui.py

## Architecture
Streamlit UI → Flask API → OCR Router → CNN / CRNN Model

-----------
