import os
from dotenv import load_dotenv
from supabase import create_client, Client
from datetime import date

# --- Supabase Connection ---

# Q1: 
# supabase-py needs your Project URL and anon (public) key. 
# You find them in the Supabase Dashboard under Project Settings -> API. 
# They should never be hardcoded because bots actively scan public GitHub repos for exposed keys to steal data or hijack cloud resources.

# Q2:
def get_client() -> Client:
    load_dotenv()
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    
    if not url or not key:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY in environment variables.")
        
    return create_client(url, key)

# Q3:
# Row Level Security (RLS) restricts database access on a per-row level based on user authentication.
# We disabled it here so our simple Python script can read/write data without jumping through complex authentication hoops.
# In a real-world application, like a multi-user SaaS platform, you would keep it enabled so that a logged-in user can only query their own private data and not the data of other users.


# --- supabase-py CRUD ---

# Q1:
def insert_test_record(supabase: Client):
    record = {
        "date": str(date.today()),
        "temperature_2m_max": 25.5,
        "temperature_2m_min": 15.0,
        "precipitation_sum": 0.0,
        "wind_speed_10m_max": 12.5
    }
    response = supabase.table("weather_raw").insert(record).execute()
    print("Insert Response:", response.data)

# COMMENT: If ran twice, the database would throw a unique constraint violation error because 'date' is our primary key. 
# To make it safe to run multiple times, you would change `.insert()` to `.upsert()`.

# Q2:
def get_records_by_date_range(supabase: Client, start: str, end: str):
    response = supabase.table("weather_raw").select("*").gte("date", start).lte("date", end).execute()
    return response.data

# Q3:
# 'insert' adds a new row and will crash if a primary key collision occurs. 'upsert' tries to insert, but if the primary key already exists, it updates the existing row instead.
# Example: Use 'insert' when a new user signs up (you want it to fail if the email is already taken). Use 'upsert' in an automated data pipeline so that rerunning it just overwrites old data instead of crashing.

def safe_upsert(supabase: Client, records: list):
    response = supabase.table("weather_raw").upsert(records, on_conflict="date").execute()
    print(f"Upserted {len(response.data)} rows.")


# --- Idempotency ---

# Q1:
# Idempotency means that executing an operation multiple times produces the exact same result as executing it once. 
# It is critical for data pipelines because scripts often fail or are manually restarted. If a non-idempotent pipeline processes 500 out of 1,000 rows, crashes, and is restarted, you'll end up with 500 duplicate rows ruining your database. An idempotent pipeline will just cleanly overwrite the first 500 and finish the rest.