# app_risk.py
import requests, datetime, math
import numpy as np

def get_coords_for_city_openweather(city, api_key):
    try:
        if not api_key:
            return None, None
        url = "http://api.openweathermap.org/geo/1.0/direct"
        r = requests.get(url, params={'q': city, 'limit':1, 'appid': api_key}, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data:
            return float(data[0]['lat']), float(data[0]['lon'])
    except:
        return None, None

def fetch_open_meteo_daily_precip(lat, lon, days=30, timezone='UTC'):
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {'latitude': lat, 'longitude': lon, 'daily':'precipitation_sum', 'forecast_days': int(days), 'timezone': timezone}
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        data = r.json()
        daily = data.get('daily', {})
        times = daily.get('time', [])
        precs = daily.get('precipitation_sum', [])
        return list(zip(times, precs))
    except:
        return []

def fetch_open_meteo_monthly_climatology(lat, lon):
    try:
        url = "https://climate-api.open-meteo.com/v1/climate"
        params = {'latitude': lat, 'longitude': lon, 'start_year':1991, 'end_year':2020, 'monthly':'precipitation_sum'}
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        data = r.json()
        monthly = data.get('monthly', {})
        precip = monthly.get('precipitation_sum', [])
        if precip and len(precip) >= 12:
            return {i+1: precip[i] for i in range(12)}
    except:
        return {}

def compute_30day_heavy_rain_probability(lat, lon, heavy_threshold_mm=50.0, prefer_open_meteo=True):
    days_needed = 30
    forecast_days = fetch_open_meteo_daily_precip(lat, lon, days=days_needed) if prefer_open_meteo else []
    forecast_count = len(forecast_days)
    forecast_heavy = sum(1 for _,p in forecast_days if p is not None and p >= heavy_threshold_mm)
    if forecast_count >= days_needed:
        prob_forecast = 1.0 if forecast_heavy>0 else 0.0
        return prob_forecast, {'forecast_days':forecast_count, 'forecast_heavy_days':forecast_heavy}
    remaining_days = max(0, days_needed - forecast_count)
    monthly = fetch_open_meteo_monthly_climatology(lat, lon)
    if not monthly:
        p_daily = 0.02
    else:
        today = datetime.date.today()
        p_vals = []
        for i in range(remaining_days):
            day = today + datetime.timedelta(days=forecast_count + i)
            month = day.month
            monthly_mm = monthly.get(month, 0.0)
            days_in_month = 30.0
            daily_mean = monthly_mm / max(1, days_in_month)
            if daily_mean <= 0:
                p = 0.0
            else:
                p = math.exp(-heavy_threshold_mm / (daily_mean + 1e-6))
                p = min(1.0, max(0.0,p))
            p_vals.append(p)
        p_daily = float(np.mean(p_vals)) if p_vals else 0.02
    p_no_heavy_remaining = (1.0 - p_daily) ** remaining_days if remaining_days>0 else 1.0
    prob_remain_at_least_one = 1.0 - p_no_heavy_remaining
    prob_forecast = 1.0 if forecast_heavy>0 else 0.0
    combined = 1.0 - (1.0 - prob_forecast) * (1.0 - prob_remain_at_least_one)
    details = {'forecast_days':forecast_count, 'forecast_heavy_days':forecast_heavy, 'remaining_days':remaining_days, 'p_daily_est':p_daily}
    return combined, details
