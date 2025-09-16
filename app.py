import streamlit as st
import joblib
import numpy as np
import os
import json

# Load model artifacts
model_dir = "./model_artifacts"
model = joblib.load(os.path.join(model_dir, "model.joblib"))
scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
le = joblib.load(os.path.join(model_dir, "label_encoder.joblib"))

with open(os.path.join(model_dir, "meta.json"), "r") as f:
    meta = json.load(f)

st.set_page_config(page_title="AI Crop Recommendation", page_icon="🌱", layout="centered")

# Sidebar for language selection
language = st.sidebar.selectbox("🌐 Language / भाषा चुनें", ["English", "Hindi"])

# Translations for UI labels
translations = {
    "title": {"English": "🌱 AI-Powered Crop Recommendation System", "Hindi": "🌱 एआई आधारित फसल सिफारिश प्रणाली"},
    "desc": {"English": "Enter soil and weather details below to get the best crop recommendation.",
             "Hindi": "नीचे मिट्टी और मौसम का विवरण दर्ज करें ताकि सर्वश्रेष्ठ फसल की सिफारिश मिले।"},
    "nitrogen": {"English": "Nitrogen (N)", "Hindi": "नाइट्रोजन (N)"},
    "phosphorus": {"English": "Phosphorous (P)", "Hindi": "फॉस्फोरस (P)"},
    "potassium": {"English": "Potassium (K)", "Hindi": "पोटैशियम (K)"},
    "temperature": {"English": "Temperature (°C)", "Hindi": "तापमान (°C)"},
    "humidity": {"English": "Humidity (%)", "Hindi": "नमी (%)"},
    "soil_ph": {"English": "Soil pH", "Hindi": "मिट्टी का pH"},
    "rainfall": {"English": "Rainfall (mm)", "Hindi": "वर्षा (मिमी)"},
    "recommend_btn": {"English": "🔍 Recommend Crop", "Hindi": "🔍 फसल की सिफारिश करें"},
    "recommended_crop": {"English": "🌾 Recommended Crop:", "Hindi": "🌾 अनुशंसित फसल:"},
    "confidence": {"English": "✅ Confidence:", "Hindi": "✅ भरोसा:"},
    "accuracy": {"English": "(Model test accuracy: {:.2f}%)", "Hindi": "(मॉडल परीक्षण सटीकता: {:.2f}%)"},
}

# Crop translations
crop_translations = {
    "rice": "चावल", "wheat": "गेहूँ", "maize": "मक्का", "chickpea": "चना",
    "kidneybeans": "राजमा", "pigeonpeas": "अरहर", "mothbeans": "मटकी", "mungbean": "मूंग",
    "blackgram": "उड़द", "lentil": "मसूर", "pomegranate": "अनार", "banana": "केला",
    "mango": "आम", "grapes": "अंगूर", "watermelon": "तरबूज", "muskmelon": "खरबूजा",
    "apple": "सेब", "orange": "संतरा", "papaya": "पपीता", "coconut": "नारियल",
    "cotton": "कपास", "jute": "जूट", "coffee": "कॉफ़ी",
}

# UI
st.title(translations["title"][language])
st.write(translations["desc"][language])

# Input fields (translated)
N = st.number_input(translations["nitrogen"][language], min_value=0, max_value=150, value=90)
P = st.number_input(translations["phosphorus"][language], min_value=0, max_value=150, value=42)
K = st.number_input(translations["potassium"][language], min_value=0, max_value=150, value=43)
temperature = st.number_input(translations["temperature"][language], min_value=0, max_value=50, value=20)
humidity = st.number_input(translations["humidity"][language], min_value=0, max_value=100, value=82)
ph = st.number_input(translations["soil_ph"][language], min_value=0.0, max_value=14.0, value=6.5, step=0.1)
rainfall = st.number_input(translations["rainfall"][language], min_value=0, max_value=500, value=202)

if st.button(translations["recommend_btn"][language]):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    crop = le.inverse_transform([prediction])[0]

    probs = model.predict_proba(features_scaled)[0]
    confidence = np.max(probs)

    crop_display = crop if language == "English" else crop_translations.get(crop, crop)

    st.success(f"{translations['recommended_crop'][language]} {crop_display.capitalize()}")
    st.info(f"{translations['confidence'][language]} {confidence*100:.2f}%")
    st.caption(translations["accuracy"][language].format(meta['accuracy']*100))
