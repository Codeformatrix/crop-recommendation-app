"""
data_builders/synthesize_local_dataset.py
- Loads Kaggle 'Crop_recommendation.csv'
- Loads climate_soil_summary.csv (from fetch_climate_and_soil.py)
- Synthesizes a localized Kaggle-style dataset by sampling rows and replacing
  climate and pH fields with district-specific draws.
- Output: data/localized_kaggle_bhopal_synth.csv
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm

KAGGLE_CSV = "Crop_recommendation.csv"   # ensure this file exists in repo root
CLIMATE_SOIL_CSV = "data/climate_soil_summary.csv"
OUT_DIR = "data"
OUT_CSV = os.path.join(OUT_DIR, "localized_kaggle_bhopal_synth.csv")
SYNTH_SIZE = 1200  # target synthetic rows (you can increase)

os.makedirs(OUT_DIR, exist_ok=True)

def load_kaggle(path):
    df = pd.read_csv(path)
    # normalize column names to lower-case
    df.columns = [c.strip() for c in df.columns]
    # typical Kaggle dataset has columns: N,P,K,temperature,humidity,ph,rainfall,label
    return df

def district_sampler(climate_df, district):
    row = climate_df[climate_df['district'] == district]
    if row.empty:
        return None
    # get means
    tmean = float(row.iloc[0]['t2m_mean']) if pd.notnull(row.iloc[0]['t2m_mean']) else None
    prmean = float(row.iloc[0]['annual_precip_mean']) if pd.notnull(row.iloc[0]['annual_precip_mean']) else None
    phprior = float(row.iloc[0]['soil_ph_prior']) if pd.notnull(row.iloc[0]['soil_ph_prior']) else None

    def sample_once():
        # sample temperature: normal around mean, small sd
        t = None
        if tmean is not None:
            t = float(np.round(np.random.normal(loc=tmean, scale=max(0.5, abs(tmean)*0.03)), 2))
        # sample rainfall: use gamma-ish variation but simple normal around mean
        r = None
        if prmean is not None:
            r = float(np.round(np.random.normal(loc=prmean, scale=max(5, abs(prmean)*0.08)), 2))
            if r < 0:
                r = float(abs(r))
        # sample pH around soil prior
        p = None
        if phprior is not None:
            p = float(np.round(np.random.normal(loc=phprior, scale=0.4), 2))
            p = float(np.clip(p, 4.5, 9.5))
        return t, r, p
    return sample_once

def synthesize(dataset, climate_df, districts, out_path, size=SYNTH_SIZE):
    synth_rows = []
    for i in tqdm(range(size)):
        row = dataset.sample(1).iloc[0].copy()
        # choose a target district randomly
        district = np.random.choice(districts)
        sampler = district_sampler(climate_df, district)
        if sampler is None:
            # fallback: leave original values
            t, r, p = None, None, None
        else:
            t, r, p = sampler()
        # map column names (best effort)
        # Kaggle column names vary; we try common ones
        if 'temperature' in row.index and t is not None:
            row['temperature'] = t
        elif 'temp' in row.index and t is not None:
            row['temp'] = t
        if 'rainfall' in row.index and r is not None:
            row['rainfall'] = r
        if 'ph' in row.index and p is not None:
            row['ph'] = p
        # attach district and lat/lon from climate_df for traceability
        cs = climate_df[climate_df['district'] == district].iloc[0]
        row['district'] = district
        row['district_lat'] = cs['lat']
        row['district_lon'] = cs['lon']
        synth_rows.append(row)
    synth_df = pd.DataFrame(synth_rows)
    synth_df.to_csv(out_path, index=False)
    print("Saved synthetic localized dataset:", out_path)
    return synth_df

if __name__ == "__main__":
    print("Loading Kaggle dataset:", KAGGLE_CSV)
    if not os.path.exists(KAGGLE_CSV):
        raise SystemExit(f"Missing {KAGGLE_CSV} in repo root. Add the Kaggle file and re-run.")
    kag = load_kaggle(KAGGLE_CSV)
    print("Loading climate/soil summary:", CLIMATE_SOIL_CSV)
    if not os.path.exists(CLIMATE_SOIL_CSV):
        raise SystemExit(f"Missing {CLIMATE_SOIL_CSV} - run fetch_climate_and_soil.py first.")
    cs = pd.read_csv(CLIMATE_SOIL_CSV)
    districts = cs['district'].tolist()
    out = synthesize(kag, cs, districts, OUT_CSV, size=SYNTH_SIZE)
    print("Sample rows:")
    print(out.head())
