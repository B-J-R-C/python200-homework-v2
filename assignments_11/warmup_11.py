from prefect import task, get_run_logger

# --- Prefect Orchestration ---

# Q1: 
# A @flow is the main container for your pipeline that manages state and orchestrates execution, while a @task is a discrete, individual unit of work (like hitting an API or querying a database). 
# You should NOT decorate a simple Celsius-to-Fahrenheit helper function with @task. Prefect adds network and state-tracking overhead to tasks. For pure, fast, in-memory math with no I/O, standard Python functions are much faster and more efficient.

# Q2:
# @task(retries=3, retry_delay_seconds=30)

# Q3:
# You would look at the "Logs" tab and the task run graph for that specific flow run in the Prefect UI (localhost:4200). 
# You would expect to find the exact Python stack trace showing why 'transform' failed (e.g., an OpenAI API timeout, a missing file path, or a KeyError), which explains why 'load_enriched' was left in a Pending/Cancelled state.


# --- Production Patterns ---

# Q1:
# `raise_for_status()` checks the HTTP response and immediately throws an HTTPError exception if the status code is 4xx or 5xx. 
# This is better than just printing "error" because printing fails silently—the script keeps running, tries to parse an error page as JSON, and crashes confusingly downstream. With `raise_for_status()`, if a 500 error occurs, the task immediately fails, triggers a Prefect retry, and if it fails again, downstream tasks are safely cancelled.

# Q2:
# `upsert` protects you from Primary Key collisions. If the pipeline crashes halfway through, some records were already inserted. 
# If you used a plain `insert` and restarted, the database would throw a 409 Conflict / 500 Error because the dates already exist, crashing the pipeline again. `upsert` makes the action idempotent by safely overwriting the existing rows.

# Q3:
@task
def log_enrichment_count(enrichment_records: list):
    logger = get_run_logger()
    logger.info(f"Successfully prepared {len(enrichment_records)} records for upsert.")

# Q4:
# The incremental check contributes to idempotency by ensuring that a rerun only processes the delta (new data) rather than blindly re-processing everything.
# If we removed it and processed all 365 records every time, the data correctness would still be fine (thanks to upsert overwriting it), but the practical consequences would be terrible: you would waste minutes of execution time and spend $0.05+ in OpenAI API tokens every single time the pipeline ran.