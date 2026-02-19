# -*- coding: utf-8 -*-
"""
Created on Tue Dec 30 19:10:27 2025

@author: Admin
"""
import streamlit as st
import requests
from PIL import Image
import base64

# ---------------- CONFIG ----------------
API_URL = "http://127.0.0.1:5000/ocr"
BG_IMAGE_PATH = r"D:\SEM III\IMAGE & VIDEO\Images\ancient.jfif"

st.set_page_config(
    page_title="IndiScan – Handwritten OCR",
    layout="centered"
)

# ---------------- SESSION STATE ----------------
if "ocr_text" not in st.session_state:
    st.session_state.ocr_text = ""

# ---------------- BACKGROUND ----------------
def bg_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg = bg_base64(BG_IMAGE_PATH)

st.markdown(f"""
<style>
.stApp {{
    background-image:
        linear-gradient(rgba(255,248,230,0.85), rgba(255,248,230,0.85)),
        url("data:image/jpg;base64,{bg}");
    background-size: cover;
}}

.header {{
    background:#3a2416;
    color:#f5e6c8;
    padding:18px;
    border-radius:14px;
    text-align:center;
    box-shadow:0 6px 18px rgba(0,0,0,0.3);
}}

.card {{
    background:#fff;
    padding:16px;
    border-radius:12px;
    box-shadow:0 4px 14px rgba(0,0,0,0.18);
}}

.ocr-box {{
    background:#3a2416;
    color:#fff2d8;
    font-size:30px;
    font-weight:700;
    padding:14px;
    border-radius:12px;
    text-align:center;
    min-height:80px;
}}

.stButton>button {{
    background:#5a3a23;
    color:white;
    font-weight:600;
    border-radius:10px;
}}
</style>
""", unsafe_allow_html=True)

# ---------------- UI ----------------

st.markdown("""
<div class="header">
<h2>✍🏻 IndiScan – Handwritten OCR</h2>
<p>Character & Word OCR for handwritten Devanagari (CPU-only)</p>
</div>
""", unsafe_allow_html=True)

st.write("")

uploaded = st.file_uploader(
    "📜 Upload Manuscript Image (JPG / PNG)",
    type=["jpg","jpeg","png"]
)

# ---------------- AFTER UPLOAD ----------------
if uploaded:
    img = Image.open(uploaded)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🖼 Uploaded Image")
        st.image(img, width=180)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🔎 OCR Output")

        if st.session_state.ocr_text:
            st.markdown(
                f'<div class="ocr-box">{st.session_state.ocr_text}</div>',
                unsafe_allow_html=True
            )
        else:
            st.caption("No output yet")

        st.markdown('</div>', unsafe_allow_html=True)

    def run_ocr(mode):
        with st.spinner("Running OCR..."):
            try:
                res = requests.post(
                    API_URL,
                    files={"image": uploaded.getvalue()},
                    data={"mode": mode},
                    timeout=30
                )
                if res.status_code == 200:
                    st.session_state.ocr_text = res.json().get("ocr_output","")
                else:
                    st.session_state.ocr_text = "OCR failed"
            except:
                st.session_state.ocr_text = "Backend not running"

    st.write("")
    b1, b2 = st.columns(2)
    with b1:
        st.button("Character OCR", use_container_width=True,
                  on_click=lambda: run_ocr("char"))
    with b2:
        st.button("Word OCR", use_container_width=True,
                  on_click=lambda: run_ocr("word"))

st.write("")
st.caption("IndiScan • MSc Mini Project • Handwritten Indic OCR")
