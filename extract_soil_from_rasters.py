import os
import rasterio
from rasterio.sample import sample_gen
import pandas as pd

# --- CONFIG ---
RASTER_DIR = "soil_rasters"
DISTRICTS = {
    "Bhopal": (23.258486, 77.401989),
    "Sehore": (23.115004, 77.066536),
    "Raisen": (23.267591, 78.172713),
    "Vidisha": (23.846300, 77.837021),
    "Rajgarh": (23.871671, 76.774470),
    "Narmadapuram": (22.600283, 77.926985)
}
PROPERTIES = ["phh2o", "soc", "clay"]
DEPTH = "0-5cm"
OUT_CSV = "data/climate_soil_summary_local.csv"

os.makedirs("data", exist_ok=True)

# --- LOAD RASTERS ---
rasters = {}
for prop in PROPERTIES:
    path = os.path.join(RASTER_DIR, f"{prop}_{DEPTH}.tif")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run download script first.")
    rasters[prop] = rasterio.open(path)

# --- EXTRACT VALUES ---
records = []
for d, (lat, lon) in DISTRICTS.items():
    record = {"district": d, "lat": lat, "lon": lon}
    for prop, raster in rasters.items():
        # raster.sample expects (lon, lat)
        val = list(raster.sample([(lon, lat)]))[0][0]
        record[f"soil_{prop}_local"] = float(val) if val is not None else None
    records.append(record)

# --- SAVE CSV ---
df = pd.DataFrame(records)
df.to_csv(OUT_CSV, index=False)
print(f"✅ Saved soil values to: {OUT_CSV}")
print(df)
