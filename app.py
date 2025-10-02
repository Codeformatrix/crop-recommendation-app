# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
from utils import load_model_artifacts, fetch_soilgrids  # keep utils.py from earlier
from app_risk import compute_30day_heavy_rain_probability, get_coords_for_city_openweather

import os
import streamlit as st

OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY", "")
if not OPENWEATHER_KEY:
    st.error("⚠️ Please set your OPENWEATHER_KEY environment variable.")
else:
    st.success("✅ OpenWeather API key loaded successfully.")


# ----------------------
# Load model and metadata
# ----------------------
MODEL_DIR = "./model_artifacts"
model, scaler, le, meta = load_model_artifacts(MODEL_DIR)
FEATURES = meta.get('features', ['N','P','K','temperature','humidity','ph','rainfall'])

# ----------------------
# Language strings (en, hi, ur)
# Add or edit keys here to control UI text.
# ----------------------
STRINGS = {
    'language_label': {'en': "Language", 'hi': "भाषा", 'ur': "زبان"},
    'app_title': {'en': "Kisan AI Sahayak", 'hi': "किसान AI सहायक", 'ur': "کسان AI مددگار"},
    'app_sub': {'en': "Simple crop advice with rainfall-risk checks", 'hi': "सरल बुवाई सलाह + वर्षा जोखिम", 'ur': "سادہ فصل مشورہ + بارش کا خطرہ"},
    'soil_fert': {'en': "Soil fertility", 'hi': "मिट्टी की उर्वरकता", 'ur': "مٹی کی زرخیزی"},
    'ph_label': {'en': "Soil pH", 'hi': "मिट्टी का pH", 'ur': "مٹی کا pH"},
    'city_label': {'en': "Nearest city / village", 'hi': "नज़दीकी शहर / गांव", 'ur': "قریب ترین شہر / گاؤں"},
    'get_advice': {'en': "Get Recommendation", 'hi': "सलाह लें", 'ur': "مشورہ لیں"},
    'risk_low': {'en': "Low risk — Safe to sow", 'hi': "कम जोखिम — बोआई सुरक्षित", 'ur': "کم خطرہ — بوتائی محفوظ"},
    'risk_moderate': {'en': "Moderate risk — Stagger sowing", 'hi': "मध्यम जोखिम — stagger बुवाई", 'ur': "درمیانہ خطرہ — بونا تقسیم کریں"},
    'risk_high': {'en': "High risk — Delay or choose tolerant crop", 'hi': "उच्च जोखिम — देरी या सहनशील फसल", 'ur': "زیادہ خطرہ — تاخیر یا برداشت کرنے والی فصل"},
    'risk_very_high': {'en': "Very High — Do not sow now", 'hi': "बहुत उच्च — अभी न बोएँ", 'ur': "بہت زیادہ — ابھی نہ بوئیں"},
    'recommended_crop': {'en': "Recommended crop", 'hi': "सुझावित फसल", 'ur': "تجویز کردہ فصل"},
    'confidence': {'en': "Confidence", 'hi': "विश्वास", 'ur': "اعتماد"},
    'support_desk': {'en': "Support Desk", 'hi': "सहायता डेस्क", 'ur': "سپورٹ ڈیسک"},
    'community_hub': {'en': "Community Hub", 'hi': "क्लब (समुदाय)", 'ur': "کمیونٹی حب"},
    'mandi_prices': {'en': "Mandi Prices", 'hi': "मंडी कीमतें", 'ur': "منڈی قیمتیں"},
    # Add more strings as needed
}

# ----------------------
# Sidebar: language, navigation, keys
# ----------------------
st.sidebar.title("⚙️ Settings")
lang_sel = st.sidebar.selectbox(STRINGS['language_label']['en'] + " / " + STRINGS['language_label']['hi'],
                                options=["English", "Hindi - हिन्दी", "Urdu - اُردُو"])
# map to code
if lang_sel.startswith("Hindi"):
    lang = 'hi'
elif lang_sel.startswith("Urdu"):
    lang = 'ur'
else:
    lang = 'en'

st.sidebar.markdown("---")
st.sidebar.info("This demo focuses on Bhopal & nearby languages (Hindi/Urdu).")

# Navigation pages in requested order
pages = [
    "1. Crop Recommendation",
    "2. Weather",
    "3. Rainfall Forecast (30d Risk)",
    "4. Mandi Price",
    "5. Support Desk",
    "6. Community Hub"
]
page = st.sidebar.radio("Go to", pages)

# Helper translation
def T(key):
    return STRINGS.get(key, {}).get(lang, STRINGS.get(key, {}).get('en', key))

# Header
st.title(T('app_title'))
st.write(T('app_sub'))

# ----------------------
# PAGE: 1 - Crop Recommendation
# ----------------------
# --- inside app.py, replace the Crop Recommendation page code with this function/page ---
def page_crop():
    import json, streamlit as st
    st.header("1. " + T('recommended_crop'))

    # Location: default auto-detect
    auto_loc = st.checkbox("Auto-detect my location (recommended)", value=True)
    lat = lon = None
    if auto_loc:
        # same JS approach you had earlier - re-use components block from your app
        st.info("Allow browser location permission (if running in a browser).")
        import streamlit.components.v1 as components
        components.html(
            """
            <script>
            navigator.geolocation.getCurrentPosition(
                (pos) => {
                    const data = {"lat": pos.coords.latitude, "lon": pos.coords.longitude};
                    window.parent.postMessage({isStreamlitMessage: true, type: "streamlit:setComponentValue", value: JSON.stringify(data)}, "*");
                },
                (err) => {
                    window.parent.postMessage({isStreamlitMessage: true, type: "streamlit:setComponentValue", value: JSON.stringify({"error": err.message})}, "*");
                });
            </script>
            """, height=0
        )
        geo_raw = st.session_state.get("component_value")
        if geo_raw:
            try:
                geo = json.loads(geo_raw)
                lat, lon = geo.get("lat"), geo.get("lon")
                st.success(f"Location detected: {lat:.3f}, {lon:.3f}")
            except:
                st.warning("Location not available from browser. Use manual mode.")
    if not lat:
        city = st.text_input(T('city_label'), value="Bhopal")
        if st.button("Resolve location"):
            lat, lon = get_coords_for_city_openweather(city, os.getenv("OPENWEATHER_KEY",""))
            if not lat:
                st.error("Could not resolve city. Enter coordinates manually.")
            else:
                st.success(f"Resolved {city} -> {lat:.3f}, {lon:.3f}")

    # Soil & fertility
    st.subheader("Soil and Fertility")
    col1, col2 = st.columns([2,1])
    with col2:
        fertility = st.radio(T('soil_fert'), ["Low", "Medium", "High"])
    with col1:
        # auto-fetch soil pH if lat/lon available
        if lat and lon:
            sg = fetch_soilgrids_local(lat, lon)
            ph_val = sg.get("phh2o")
            if ph_val is not None:
                st.markdown(f"**Auto-detected soil pH:** {ph_val:.2f}")
            else:
                ph_val = st.slider(T('ph_label'), 3.0, 9.0, 6.5, step=0.1)
        else:
            ph_val = st.slider(T('ph_label'), 3.0, 9.0, 6.5, step=0.1)

    # map fertility to NPK (same mapping you used)
    if fertility.startswith("Low"):
        N,P,K = 20.0, 10.0, 10.0
    elif fertility.startswith("Medium"):
        N,P,K = 60.0, 25.0, 25.0
    else:
        N,P,K = 90.0, 42.0, 43.0

    # fetch weather (OpenWeather) to fill temperature/humidity/rainfall
    OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY","")
    weather = None
    if lat and lon and OPENWEATHER_KEY:
        weather = fetch_openweather(lat, lon, OPENWEATHER_KEY)

    temp = weather['avg_temp'] if weather and weather.get('avg_temp') is not None else st.number_input("Temperature (°C)", value=25.0)
    hum = weather['avg_humidity'] if weather and weather.get('avg_humidity') is not None else st.slider("Humidity (%)", 10, 100, 60)
    rain7 = weather['next_7d_rainfall_mm'] if weather else 0.0

    # Build model input
    FEATURES = meta.get('features', ['N','P','K','temperature','humidity','ph','rainfall'])
    values_map = {'N': N, 'P': P, 'K': K, 'temperature': temp, 'humidity': hum, 'ph': ph_val, 'rainfall': rain7}
    input_df = pd.DataFrame([ {f: values_map.get(f,0.0) for f in FEATURES} ])

    st.markdown("### Ready to predict")
    if st.button(T('get_advice')):
        if model is None:
            st.error("Model artifact not found. Place `crop_recommender.pkl` in model_artifacts/")
            return
        # scale -> predict
        try:
            X = scaler.transform(input_df) if scaler else input_df.values
            probs = model.predict_proba(X)[0]
            idx = probs.argmax()
            crop = le.inverse_transform([idx])[0] if le else str(idx)
            conf = probs[idx]
        except Exception as e:
            st.error("Prediction failed: " + str(e))
            return

        # Show result + simple "why" (top features via feature_importances if classifier supports)
        st.success(f"✅ Recommended: {crop}  —  Confidence: {conf*100:.1f}%")
        # quick explanation: top 3 features using model.feature_importances_ if available
        try:
            import numpy as np
            if hasattr(model, 'feature_importances_'):
                fi = np.array(model.feature_importances_)
                topk = fi.argsort()[-3:][::-1]
                explanations = [f"{FEATURES[i]} (importance {fi[i]:.3f})" for i in topk]
                st.write("Top factors:", ", ".join(explanations))
        except Exception:
            pass

        # Simple farmer-friendly advice (template)
        advice = f"Based on soil pH {ph_val:.1f}, expected weather, and soil fertility, we recommend planting {crop}. Use recommended seed rate and ensure proper sowing time."
        st.info(advice)


# PAGE: 2 - Weather (short)
# ----------------------
def page_weather():
    st.header("2. Weather")
    st.write("Short-term weather (7-day) and basic stats. (Uses OpenWeather / fetch_openweather).")
    city = st.text_input("City for weather", value="Bhopal")
    if st.button("Fetch weather now"):
        lat, lon = get_coords_for_city_openweather(city, os.getenv("OPENWEATHER_KEY",""))
        if lat is None:
            st.error("Could not resolve city. Enter API key or use correct city name.")
            return
        from utils import fetch_openweather
        w = fetch_openweather(lat, lon, os.getenv("OPENWEATHER_KEY",""))
        st.json(w)

# ----------------------
# PAGE: 3 - Rainfall Forecast (30d)
# ----------------------
def page_rainfall():
    st.header("3. Rainfall Forecast (30-day risk)")
    city = st.text_input("City for 30-day forecast", value="Bhopal")
    if st.button("Compute 30-day risk"):
        lat, lon = get_coords_for_city_openweather(city, os.getenv("OPENWEATHER_KEY",""))
        prob30, details = compute_30day_heavy_rain_probability(lat, lon)
        st.metric("Chance ≥1 heavy day (30d)", f"{prob30*100:.1f}%")
        st.json(details)

# ----------------------
# PAGE: 4 - Mandi Prices (placeholder)
# ----------------------
def page_mandi():
    st.header("4. Mandi Price (demo)")
    st.write("This page will fetch Agmarknet / local mandi data. For demo we show placeholders.")
    st.table(pd.DataFrame([{"crop":"Rice","price_rs_qtl":2200}, {"crop":"Wheat","price_rs_qtl":1850}]))

# ----------------------
# PAGE: 5 - Support Desk (simple)
# ----------------------
def page_support():
    st.header("5. Support Desk")
    st.write("Submit an issue or question. This demo saves to local file `support_messages.json`.")
    name = st.text_input("Your name")
    phone = st.text_input("Phone (optional)")
    msg = st.text_area("Message / समस्या विवरण")
    if st.button("Send"):
        entry = {"name": name, "phone": phone, "message": msg, "city": st.session_state.get('city', 'unknown')}
        p = "support_messages.json"
        data = []
        if os.path.exists(p):
            try:
                data = pd.read_json(p).to_dict(orient='records')
            except:
                data = []
        data.append(entry)
        pd.DataFrame(data).to_json(p, orient='records', indent=2)
        st.success("Message saved. Local demo only.")

# ----------------------
# PAGE: 6 - Community Hub (simple message board)
# ----------------------
def page_community():
    st.header("6. Community Hub")
    st.write("Farmers can post short messages. This demo stores messages locally (JSON).")
    user = st.text_input("Name")
    post = st.text_input("Post (max 200 chars)")
    if st.button("Post"):
        msg_file = "community_posts.json"
        posts = []
        if os.path.exists(msg_file):
            try:
                posts = pd.read_json(msg_file).to_dict(orient='records')
            except:
                posts = []
        posts.insert(0, {"name": user, "post": post})
        pd.DataFrame(posts).to_json(msg_file, orient='records', indent=2)
        st.success("Posted (demo).")
    # Show latest posts
    if os.path.exists("community_posts.json"):
        try:
            posts = pd.read_json("community_posts.json").to_dict(orient='records')
            for p in posts[:20]:
                st.write(f"**{p.get('name','')}** — {p.get('post','')}")
        except:
            st.info("No posts or reading error.")

# ----------------------
# Page router
# ----------------------
if page.startswith("1."):
    page_crop()
elif page.startswith("2."):
    page_weather()
elif page.startswith("3."):
    page_rainfall()
elif page.startswith("4."):
    page_mandi()
elif page.startswith("5."):
    page_support()
else:
    page_community()
