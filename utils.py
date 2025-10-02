# utils.py
import os, joblib, rasterio, requests, math
import pandas as pd

# --- model loader (expects joblib artifacts in model_artifacts/) ---
def load_model_artifacts(model_dir="./model_artifacts"):
    """
    Return (model, scaler, label_encoder, meta) or raise helpful errors.
    """
    # Adjust filenames if your repo uses different ones
    mpath = os.path.join(model_dir, "crop_recommender.pkl")
    spath = os.path.join(model_dir, "scaler.pkl")
    lpath = os.path.join(model_dir, "label_encoder.pkl")
    meta_path = os.path.join(model_dir, "meta.json")

    model = joblib.load(mpath) if os.path.exists(mpath) else None
    scaler = joblib.load(spath) if os.path.exists(spath) else None
    le = joblib.load(lpath) if os.path.exists(lpath) else None
    meta = {}
    if os.path.exists(meta_path):
        try:
            meta = pd.read_json(meta_path).to_dict()
        except:
            import json
            with open(meta_path) as f:
                meta = json.load(f)
    return model, scaler, le, meta

# --- local SoilGrids raster lookup ---
RASTER_FILES = {
    "phh2o": "soil_rasters/phh2o_0-5cm.tif",
    "soc": "soil_rasters/soc_0-5cm.tif",
    "clay": "soil_rasters/clay_0-5cm.tif"
}

def fetch_soilgrids_local(lat, lon):
    """Return dict with phh2o, soc, clay (floats or None)."""
    out = {}
    for prop, path in RASTER_FILES.items():
        try:
            if not os.path.exists(path):
                out[prop] = None
                continue
            with rasterio.open(path) as src:
                row, col = src.index(lon, lat)   # note: src.index expects lon,lat if CRS is EPSG:4326
                val = src.read(1)[row, col]
                if val == src.nodata:
                    out[prop] = None
                else:
                    out[prop] = float(val)
        except Exception as e:
            out[prop] = None
    return out

# --- OpenWeather helper (simple) ---
def get_coords_for_city_openweather(city, key):
    """Return (lat, lon) using OpenWeather geocoding."""
    if not key:
        return None, None
    url = "http://api.openweathermap.org/geo/1.0/direct"
    r = requests.get(url, params={"q": city, "limit": 1, "appid": key}, timeout=10)
    if r.status_code != 200:
        return None, None
    data = r.json()
    if not data:
        return None, None
    return data[0]["lat"], data[0]["lon"]

def fetch_openweather(lat, lon, key):
    """Simple 7-day weather fetch (use OneCall or current+forecast). Return small dict or None."""
    if not key:
        return None
    url = "https://api.openweathermap.org/data/2.5/onecall"
    params = {"lat": lat, "lon": lon, "exclude": "minutely,hourly", "appid": key, "units": "metric"}
    r = requests.get(url, params=params, timeout=10)
    if r.status_code != 200:
        return None
    j = r.json()
    # compute 7-day avg temp/humidity and total precipitation (if present)
    temps = [d.get("temp", {}).get("day") for d in j.get("daily", [])][:7]
    hums = [d.get("humidity") for d in j.get("daily", [])][:7]
    prec = [ (d.get("rain") or 0) + (d.get("snow") or 0) for d in j.get("daily", [])][:7]
    return {
        "avg_temp": float(sum(temps)/len(temps)) if temps else None,
        "avg_humidity": float(sum(hums)/len(hums)) if hums else None,
        "next_7d_rainfall_mm": float(sum(prec)) if prec else 0.0
    }
