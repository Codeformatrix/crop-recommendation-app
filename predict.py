print("✅ predict.py started")
import joblib
import numpy as np
import argparse
import json
import os

# Load artifacts
model_dir = "./model_artifacts"
model = joblib.load(os.path.join(model_dir, "model.joblib"))
scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
le = joblib.load(os.path.join(model_dir, "label_encoder.joblib"))

with open(os.path.join(model_dir, "meta.json"), "r") as f:
    meta = json.load(f)

def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    """
    Predict the best crop given soil & weather inputs.
    """
    # Prepare input
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    features_scaled = scaler.transform(features)

    # Predict
    prediction = model.predict(features_scaled)[0]
    crop = le.inverse_transform([prediction])[0]

    # Predict probability for confidence
    probs = model.predict_proba(features_scaled)[0]
    confidence = np.max(probs)

    return crop, round(float(confidence), 3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop Recommendation Prediction")
    parser.add_argument("--N", type=float, required=True, help="Nitrogen value")
    parser.add_argument("--P", type=float, required=True, help="Phosphorous value")
    parser.add_argument("--K", type=float, required=True, help="Potassium value")
    parser.add_argument("--temperature", type=float, required=True, help="Temperature (°C)")
    parser.add_argument("--humidity", type=float, required=True, help="Humidity (%)")
    parser.add_argument("--ph", type=float, required=True, help="Soil pH")
    parser.add_argument("--rainfall", type=float, required=True, help="Rainfall (mm)")

    args = parser.parse_args()
    crop, confidence = predict_crop(args.N, args.P, args.K, args.temperature, args.humidity, args.ph, args.rainfall)

    print(f"\n🌱 Recommended Crop: {crop}")
    print(f"✅ Confidence: {confidence * 100:.2f}%")
    print(f"(Model accuracy on test set: {meta['accuracy'] * 100:.2f}%)")
