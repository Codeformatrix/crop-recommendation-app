"""
fetch_climate_and_soil.py
- Geocodes district names (Nominatim) or uses fallback coords
- Fetches NASA POWER daily point data (T2M, PRECTOTCORR) and summarizes climatology
- Fetches SoilGrids properties locally from raster files (phh2o, soc, clay)
- Saves CSV: data/climate_soil_summary.csv
"""

import pandas as pd
import time
import os
from geopy.geocoders import Nominatim
from tqdm import tqdm
import rasterio

# --- CONFIG ---
OUT_DIR = "data"
OUT_CSV = os.path.join(OUT_DIR, "climate_soil_summary.csv")
DISTRICTS = ["Bhopal", "Sehore", "Raisen", "Vidisha", "Rajgarh", "Narmadapuram"]

FALLBACK_COORDS = {
    "Bhopal": (23.2599, 77.4126),
    "Sehore": (22.7441, 77.0886),
    "Raisen": (22.3870, 77.8183),
    "Vidisha": (23.5252, 77.8086),
    "Rajgarh": (24.0043, 76.5264),
    "Narmadapuram": (22.7468, 77.7186)
}

START = "20150101"
END = "20241231"
NASA_PARAMS = "T2M,PRECTOTCORR"  # mean temperature (°C) & precipitation (mm)

# Soil raster files
RASTER_FILES = {
    "phh2o": "soil_rasters/phh2o_0-5cm.tif",
    "soc": "soil_rasters/soc_0-5cm.tif",
    "clay": "soil_rasters/clay_0-5cm.tif"
}

# --- helpers ---
def geocode_place(place, country_hint="Madhya Pradesh, India"):
    geolocator = Nominatim(user_agent="crop-recommender-geocoder")
    q = f"{place}, {country_hint}"
    try:
        loc = geolocator.geocode(q, timeout=10)
        if loc:
            return loc.latitude, loc.longitude
    except Exception as e:
        print(f"[geocode] warning for {q}: {e}")
    return None

def fetch_nasa_power_point(lat, lon, start=START, end=END, params=NASA_PARAMS):
    import requests
    base = "https://power.larc.nasa.gov/api/temporal/daily/point"
    q = {
        "latitude": lat, "longitude": lon,
        "start": start, "end": end,
        "parameters": params, "community": "AG", "format": "JSON"
    }
    try:
        r = requests.get(base, params=q, timeout=30)
        r.raise_for_status()
        j = r.json()
        data = j.get("properties", {}).get("parameter", {})
        if not data:
            return None
        df = pd.DataFrame(data)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        print(f"[NASA] request error at {lat},{lon}: {e}")
        return None

def fetch_soilgrids_local(lat, lon):
    """
    Fetch soil properties (phh2o, soc, clay) from local SoilGrids rasters
    for a given lat/lon.
    """
    results = {}
    for prop, file in RASTER_FILES.items():
        try:
            with rasterio.open(file) as src:
                row, col = src.index(lon, lat)
                val = src.read(1)[row, col]
                if val == src.nodata:
                    val = None
                results[prop] = float(val) if val is not None else None
        except Exception as e:
            print(f"[SoilGrids Local] error for {prop} at {lat},{lon}: {e}")
            results[prop] = None
    return results

# --- main ---
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    records = []
    print("Will fetch NASA POWER & SoilGrids for these districts:", DISTRICTS)

    for d in tqdm(DISTRICTS):
        # 1) Geocode
        coord = geocode_place(d)
        if coord is None:
            coord = FALLBACK_COORDS.get(d)
            print(f"[geocode] used fallback for {d} -> {coord}")
        lat, lon = coord

        # 2) NASA POWER
        df_nasa = fetch_nasa_power_point(lat, lon)
        if df_nasa is not None:
            df_year = df_nasa.resample('YE').agg({'T2M': 'mean', 'PRECTOTCORR': 'sum'})
            t2m_mean = df_year['T2M'].mean()
            prectot_mean = df_year['PRECTOTCORR'].mean()
        else:
            t2m_mean, prectot_mean = None, None

        # 3) SoilGrids local
        sg = fetch_soilgrids_local(lat, lon)
        ph_mean = sg.get("phh2o")
        soc_mean = sg.get("soc")
        clay_mean = sg.get("clay")

        records.append({
            "district": d,
            "lat": lat,
            "lon": lon,
            "t2m_mean": t2m_mean,
            "annual_precip_mean": prectot_mean,
            "soil_ph_prior": ph_mean,
            "soil_soc_prior": soc_mean,
            "soil_clay_prior": clay_mean
        })

        time.sleep(1)  # polite pause

    df = pd.DataFrame(records)
    df.to_csv(OUT_CSV, index=False)
    print("✅ Saved climate+soil summary to:", OUT_CSV)
    print(df)

if __name__ == "__main__":
    main()
