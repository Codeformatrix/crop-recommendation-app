import requests
import pandas as pd
from tqdm import tqdm
import time

# -------------------------------
# 1. NASA POWER Climate Fetch
# -------------------------------
def fetch_climate_from_nasa(lat, lon):
    url = (
        "https://power.larc.nasa.gov/api/temporal/climatology/point"
        f"?parameters=T2M,PRECTOT&community=AG&longitude={lon}&latitude={lat}&format=JSON"
    )
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        t2m = data["properties"]["parameter"]["T2M"]
        pr = data["properties"]["parameter"]["PRECTOT"]

        return {
            "t2m_mean": sum(t2m.values()) / len(t2m),
            "annual_precip_mean": sum(pr.values())
        }
    except Exception as e:
        print(f"[NASA] Error fetching climate at {lat},{lon}: {e}")
        return {"t2m_mean": None, "annual_precip_mean": None}
print("[NASA raw response]", r.json())


# -------------------------------
# 2. Bhuvan Soil Fetch
# -------------------------------
def fetch_soil_from_bhuvan(lat, lon, property_layer):
    base_url = "https://bhuvan-app1.nrsc.gov.in/bhuvan/wfs"
    params = {
        "service": "WFS",
        "version": "1.0.0",
        "request": "GetFeature",
        "typeName": f"Soil:{property_layer}",  # example: Soil:Soil_pH_0_15cm
        "outputFormat": "application/json",
        "srsName": "EPSG:4326",
        "bbox": f"{lon},{lat},{lon},{lat}"
    }

    try:
        r = requests.get(base_url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if "features" in data and len(data["features"]) > 0:
            return data["features"][0]["properties"]
        else:
            return None
    except Exception as e:
        print(f"[Bhuvan] Error fetching {property_layer} at {lat},{lon}: {e}")
        return None


def fetch_all_soil_properties(lat, lon):
    layers = {
        "soil_ph_prior": "Soil_pH_0_15cm",
        "soil_oc_prior": "Soil_OC_0_15cm",
        "soil_n_prior": "Soil_Available_N",
        "soil_p_prior": "Soil_Available_P",
        "soil_k_prior": "Soil_Available_K"
    }
    results = {}
    for key, layer in layers.items():
        results[key] = fetch_soil_from_bhuvan(lat, lon, layer)
        time.sleep(1)  # small delay to avoid hitting API limits
    return results
print("[Bhuvan raw response]", r.text[:500])


# -------------------------------
# 3. Districts & Coordinates
# -------------------------------
districts = {
    "Bhopal": (23.2599, 77.4126),
    "Sehore": (23.1150, 77.0665),
    "Raisen": (23.2676, 78.1727),
    "Vidisha": (23.8463, 77.8370),
    "Rajgarh": (23.8717, 76.7745),
    "Narmadapuram": (22.6003, 77.9270)
}


# -------------------------------
# 4. Main Loop
# -------------------------------
rows = []
print(f"Will fetch NASA POWER & Bhuvan Soil data for these districts: {list(districts.keys())}")

for district, (lat, lon) in tqdm(districts.items(), total=len(districts)):
    # Climate
    climate = fetch_climate_from_nasa(lat, lon)

    # Soil
    soil = fetch_all_soil_properties(lat, lon)

    row = {
        "district": district,
        "lat": lat,
        "lon": lon,
        **climate,
        **soil
    }
    rows.append(row)

# -------------------------------
# 5. Save to CSV
# -------------------------------
df = pd.DataFrame(rows)
out_path = "data/climate_soil_summary.csv"
df.to_csv(out_path, index=False)
print(f"✅ Saved climate+soil summary to: {out_path}")
print(df)
