# Video Link: https://drive.google.com/file/d/1cRhOqco1O-K8f9q3oApK9OaDim8LA39j/view?usp=sharing 

import os
import requests
from dotenv import load_dotenv
from supabase import create_client, Client

# --- Setup Connection ---
load_dotenv()
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
if not url or not key:
    raise ValueError("Missing Supabase credentials in .env")

supabase: Client = create_client(url, key)

# --- Step 1: Extract ---
# Pulling 2023 historical data for Atlanta, GA
api_url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 33.74,
    "longitude": -84.38,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"],
    "timezone": "America/New_York"
}

print("Fetching data from Open-Meteo...")
response = requests.get(api_url, params=params)
response.raise_for_status()
data = response.json()
print("API Response fetched successfully!\n")


# --- Step 2: Transform ---
daily_data = data["daily"]
num_days = len(daily_data["time"])
records = []

for i in range(num_days):
    records.append({
        "date": daily_data["time"][i],
        "temperature_2m_max": daily_data["temperature_2m_max"][i],
        "temperature_2m_min": daily_data["temperature_2m_min"][i],
        "precipitation_sum": daily_data["precipitation_sum"][i],
        "wind_speed_10m_max": daily_data["wind_speed_10m_max"][i]
    })

print(f"First record: {records[0]}")
print(f"Last record: {records[-1]}\n")

# COMMENT: I expect 365 records for a non-leap year like 2023, and I got exactly 365. 
# If the numbers differed, it could be due to missing days in the API's dataset, or timezone offsets shifting a few boundary hours into an adjacent day.


# --- Step 3: Load ---
print("Upserting records to Supabase...")
upsert_response = supabase.table("weather_raw").upsert(records, on_conflict="date").execute()
print(f"Successfully upserted {len(upsert_response.data)} rows into weather_raw.\n")

# COMMENT: Running the script a second time upserts the exact same 365 rows without changing the total row count in the database. 
# This tells me our pipeline is completely idempotent—rerunning it is perfectly safe and won't bloat the database with duplicates.


# --- Step 4: Verify ---
print("--- Verification ---")
verify_response = supabase.table("weather_raw").select("date").execute()
dates = sorted([row["date"] for row in verify_response.data])

print(f"Total rows in DB: {len(dates)}")
print(f"Earliest date:    {dates[0]}")
print(f"Latest date:      {dates[-1]}")

july_4th = supabase.table("weather_raw").select("*").eq("date", "2023-07-04").execute()
print(f"July 4th Record:  {july_4th.data}")