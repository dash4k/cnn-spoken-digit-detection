import streamlit as st
import numpy as np
import os
import pickle
import scipy.io.wavfile as wav
from settings import DEMO_FILE
from model import extract_mfcc_matrix, load_model, CNN

st.set_page_config(page_title="Spoken Digit Detection (CNN)", layout="centered")

# Load model
model_data = load_model()
model = model_data['model']
label_names = model_data['label_names']
accuracy = model_data['accuracy']
precision = model_data['precision']
recall = model_data['recall']
f1 = model_data['f1']
mean = model_data.get('mean', None)
std = model_data.get('std', None)

# util
def preprocess_mfcc(signal, sr):
    mfcc = extract_mfcc_matrix(signal, sr)
    if mfcc.shape[1] < 16:
        raise ValueError("Recording too short or invalid: requires at least 16 frames.")
    mfcc = mfcc[:, :16]
    return np.expand_dims(np.expand_dims(mfcc, axis=0), axis=0)

# ui
st.title("🧠 Real-Time Spoken Digit Detection (CNN)")
st.caption("Spoken Digit Detection using a Convolutional Neural Network and MFCC")
st.divider()

# demo
st.subheader("🔊 Demo")
if os.path.exists(DEMO_FILE):
    st.audio(DEMO_FILE, format="audio/wav")
    if st.button("📊 Predict from Recording"):
        try:
            sr, signal = wav.read(DEMO_FILE)
            input_tensor = preprocess_mfcc(signal, sr)
            prediction = model.predict(input_tensor)[0]
            st.success(f"**Predicted Digit (from recording):** {prediction}")
        except Exception as e:
            st.error(f"Error: {e}")

# file upload
st.divider()
st.subheader("📁 Upload File and Predict")

uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])
if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    if st.button("📊 Predict from File"):
        try:
            sr, signal = wav.read(uploaded_file)
            input_tensor = preprocess_mfcc(signal, sr)
            prediction = model.predict(input_tensor)[0]
            st.success(f"**Predicted Digit (from file):** {prediction}")
        except Exception as e:
            st.error(f"Error: {e}")

# model's note
st.divider()
st.subheader("ℹ️ Model Notes")
st.info("CNN trained with MFCC features (20 coefficients × 16 frames), 4 conv filters, and 10 output classes.")
st.subheader("📈 Model Performance Summary")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{accuracy:.2%}")
col2.metric("Precision", f"{precision:.2%}")
col3.metric("Recall", f"{recall:.2%}")
col4.metric("F1 Score", f"{f1:.2%}")


# author
st.divider()
st.subheader("Authors")

data = [
    ["Danishwara Pracheta", "2308561050", "@dash4k"],
    ["Maliqy Numurti", "2308561068", "@Maliqytritata"],
    ["Krisna Udayana", "2308561122", "@KrisnaUdayana"],
    ["Dewa Sutha", "2308561137", "@DewaMahattama"]
]

for row in data:
    cols = st.columns(3)
    for i, col in enumerate(cols):
        with col:
            st.write(row[i])