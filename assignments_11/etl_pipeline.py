# Video Link: [Insert Link Here]

import os
import json
import requests
import joblib
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import OpenAI
from prefect import task, flow

# --- Clients Setup ---
load_dotenv()
supabase: Client = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# --- Tasks ---

@task(retries=2, retry_delay_seconds=10)
def extract() -> list:
    print("Extracting data from Open-Meteo...")
    api_url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 33.74, # Atlanta
        "longitude": -84.38,
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"],
        "timezone": "America/New_York"
    }
    
    response = requests.get(api_url, params=params)
    response.raise_for_status()
    data = response.json()
    
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
        
    print(f"Extracted {len(records)} daily records.")
    return records


@task(retries=2, retry_delay_seconds=5)
def load_raw(records: list):
    print("Upserting to weather_raw...")
    response = supabase.table("weather_raw").upsert(records, on_conflict="date").execute()
    print(f"Upserted {len(response.data)} records into weather_raw.")


@task
def transform() -> list:
    print("Starting Transform Step...")
    # Incremental check
    raw_response = supabase.table("weather_raw").select("*").execute()
    raw_records = raw_response.data
    
    enriched_response = supabase.table("weather_enriched").select("date").execute()
    enriched_dates = {row["date"] for row in enriched_response.data}
    
    unprocessed = [r for r in raw_records if r["date"] not in enriched_dates]
    if not unprocessed:
        print("No new records to process.")
        return []
        
    print(f"Processing {len(unprocessed)} new records.")
    
    # Load ML Model with absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    classifier = joblib.load(os.path.join(script_dir, "models", "weather_classifier.pkl"))
    with open(os.path.join(script_dir, "models", "weather_classifier_metadata.json"), "r") as f:
        feature_cols = json.load(f)["features"]
        
    df = pd.DataFrame(unprocessed)[feature_cols]
    predictions = classifier.predict(df)
    probabilities = classifier.predict_proba(df)
    
    # LLM Processing
    final_records = []
    system_prompt = "You are a running assistant. Given the weather features and a model's prediction, write one short sentence explaining why it is or isn't a good day to run. Do not mention the ML model."
    
    for i, row in enumerate(unprocessed):
        if (i + 1) % 50 == 0:
            print(f"LLM processed {i + 1} / {len(unprocessed)} records...")
            
        pred = bool(predictions[i])
        conf = round(float(max(probabilities[i])), 3)
        
        user_content = f"Prediction: {pred}. Features: {row}"
        
        try:
            llm_resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ]
            )
            llm_summary = llm_resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM Error on {row['date']}: {e}")
            llm_summary = "Weather context unavailable."
            
        final_records.append({
            "date": row["date"],
            "good_for_running": pred,
            "confidence": conf,
            "llm_summary": llm_summary
        })
        
    return final_records


@task(retries=2, retry_delay_seconds=5)
def load_enriched(records: list):
    if not records:
        print("No enriched records to load. Skipping.")
        return
        
    print("Upserting to weather_enriched...")
    response = supabase.table("weather_enriched").upsert(records, on_conflict="date").execute()
    print(f"Upserted {len(response.data)} records into weather_enriched.")


# --- Flow ---

@flow(log_prints=True)
def weather_etl_pipeline():
    print("Starting full ETL pipeline...")
    raw_records = extract()
    load_raw(raw_records)
    enriched_records = transform()
    load_enriched(enriched_records)
    print("Pipeline execution complete! Check the Prefect UI for details.")

if __name__ == "__main__":
    weather_etl_pipeline()