# Video Link: https://drive.google.com/file/d/1DMiA8JjEqoCG4x0HgIGRXZLQv78ae6O7/view?usp=sharing

import os
import json
import joblib
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import OpenAI

# --- Setup ---
load_dotenv()
supabase: Client = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# --- Step 1: Incremental Read ---
print("Fetching records from Supabase...")
raw_response = supabase.table("weather_raw").select("*").execute()
raw_records = raw_response.data

enriched_response = supabase.table("weather_enriched").select("date").execute()
enriched_dates = {row["date"] for row in enriched_response.data}

unprocessed_records = [row for row in raw_records if row["date"] not in enriched_dates]

print(f"Total raw records: {len(raw_records)}")
print(f"Already enriched:  {len(enriched_dates)}")
print(f"To process today:  {len(unprocessed_records)}\n")

if not unprocessed_records:
    print("No new records to process. Exiting early.")
    exit()

# --- Step 2: ML Transform ---
print("Loading ML model and running predictions...")

# Get the exact folder where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build the bulletproof paths
model_path = os.path.join(script_dir, "models", "weather_classifier.pkl")
meta_path = os.path.join(script_dir, "models", "weather_classifier_metadata.json")

classifier = joblib.load(model_path)
with open(meta_path, "r") as f:
    metadata = json.load(f)
feature_cols = metadata["features"]

# Build DataFrame ensuring column order matches training
df = pd.DataFrame(unprocessed_records)[feature_cols]

predictions = classifier.predict(df)
probabilities = classifier.predict_proba(df)

# Create a list to hold the intermediate records
enrichment_batch = []
good_count = 0
confidences = []

for i, row in enumerate(unprocessed_records):
    pred = bool(predictions[i])
    conf = round(float(max(probabilities[i])), 3) # Max probability is the confidence
    
    if pred:
        good_count += 1
    confidences.append(conf)
    
    enrichment_batch.append({
        "date": row["date"],
        "features": row,  # Keep features for the LLM prompt
        "good_for_running": pred,
        "confidence": conf
    })

print(f"ML Step Complete: {good_count} days classified as good for running.")
print(f"Confidence range: {min(confidences)} to {max(confidences)}\n")

# --- Step 3: LLM Transform ---
print("Running LLM Enrichment...")
system_prompt = (
    "You are a helpful running assistant. Given the weather features and a machine learning model's prediction on whether "
    "it is a good day to run, write exactly one short, natural-sounding sentence explaining why. "
    "Be direct. Do not say 'The machine learning model predicted'. Just give the weather reasoning."
)

final_records = []
for i, record in enumerate(enrichment_batch):
    if (i + 1) % 50 == 0:
        print(f"Processed {i + 1} / {len(enrichment_batch)} LLM calls...")
        
    user_content = (
        f"Prediction: {'Good for running' if record['good_for_running'] else 'Not good for running'} "
        f"(Confidence: {record['confidence']}).\n"
        f"Weather: Max Temp {record['features']['temperature_2m_max']}C, Min Temp {record['features']['temperature_2m_min']}C, "
        f"Precipitation {record['features']['precipitation_sum']}mm, Max Wind {record['features']['wind_speed_10m_max']}km/h."
    )
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.7
        )
        llm_summary = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error on {record['date']}: {e}")
        llm_summary = "Weather data unavailable for summary."
        
    final_records.append({
        "date": record["date"],
        "good_for_running": record["good_for_running"],
        "confidence": record["confidence"],
        "llm_summary": llm_summary
    })

print("LLM step complete.\n")

# --- Step 4: Load ---
print("Upserting enriched records to Supabase...")
upsert_response = supabase.table("weather_enriched").upsert(final_records, on_conflict="date").execute()
print(f"Successfully upserted {len(upsert_response.data)} rows.\n")

# --- Step 5: Verify ---
print("--- Verification ---")
verify_response = supabase.table("weather_enriched").select("*").execute()
data = verify_response.data

print(f"Total enriched rows: {len(data)}")
print("5 Sample Rows:")
for row in data[:5]:
    print(f"{row['date']} | Good: {row['good_for_running']} ({row['confidence']}) | {row['llm_summary']}")

total_good = sum(1 for r in data if r["good_for_running"])
print(f"\nTotal days classified as good for running: {total_good}\n")

# COMMENT: Looking at the samples, the LLM summarizes well based on the inputs. 
# One great summary highlighted the cool minimum temperatures and lack of rain making it an ideal morning run. 
# One weaker summary felt a bit contradictory, calling it "not good for running" due to heat, even though the max temp was only around 20C. This happens because the LLM is blindly trusting the ML prediction without having hardcoded temperature rules.

# --- Step 6: Reflect ---
# If the classifier was trained on Charlotte data but is predicting on data from a different city, the accuracy is inherently uncertain. Because the model was trained on one specific city's distribution, its performance may shift or degrade if the new city's weather patterns differ. In machine learning, this data distribution shift means we cannot guarantee accuracy without retraining or validating the model on the new region's data.
# 
# The LLM has zero ability to override the classifier in this architecture—it is purely additive downstream. The implications are that if the ML model is wrong, the LLM will confidently hallucinate an explanation trying to justify that incorrect prediction. 
# 
# If running this pipeline on 50,000 records, my main concern would be latency and cost from the LLM step. Making 50,000 synchronous API calls to OpenAI would take hours and rack up significant token costs. I would address this by utilizing OpenAI's Batch API to process thousands of prompts asynchronously at a 50% discount, or by rewriting the script to use async Python (`asyncio` and `aiohttp`) to fire off concurrent requests.