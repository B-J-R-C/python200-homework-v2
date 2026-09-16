import time
from openai import OpenAI

# --- ML vs. LLM in Pipelines ---

# Q1:
# The ML classifier produces a fast, cheap, binary prediction (good_for_running: True/False) and a numeric confidence score. 
# The LLM produces a natural language summary explaining the weather context.
# They do this because ML is excellent at finding statistical boundaries in numbers, while LLMs are built to synthesize and generate human-readable text.
# If we swapped them: using the LLM for binary prediction would be slow, expensive, and prone to hallucinating the wrong class. Using the ML model for recommendations would require complex, rigid Natural Language Generation (NLG) templates that lack conversational fluidity.

# Q2:
# 1. Date string to day-of-week: Deterministic code (Python's datetime library handles this perfectly without the overhead of AI).
# 2. Job posting classification: LLM (The text is unstructured, nuanced, and variable, which LLMs excel at parsing).
# 3. Predicting customer churn: Trained ML model (It's a structured, tabular data problem with known numeric features and a labeled target—perfect for a random forest or logistic regression).
# 4. Normalizing city names: Deterministic code (A simple fuzzy-matching library or dictionary mapping is much faster, cheaper, and 100% reliable compared to an LLM).
# 5. Summing revenue: Deterministic code (Basic arithmetic does not require artificial intelligence).

# Q3:
# Incremental processing means querying the target database to see what has already been processed, and only running the transform steps on new, unseen records. 
# It is critical here because LLM API calls cost money and take seconds per record. If the script re-processed all 365 records every time, it would rewrite the exact same data (no data correctness issues due to upserting), but you would waste hundreds of API calls and sit waiting for minutes on every single run.


# --- Prompt Design ---

# Q1:
# System prompt: "You are a running weather assistant. First, state the prediction clearly. Second, explain the reasoning based on the provided weather features. You must return exactly two sentences."
# Validation logic change: Instead of just checking if the response exists, you would need to split the response by periods (e.g., `response.split('.')`) and assert that it results in at least two valid sentences, prompting a retry if it only returns one.

# Q2:
def call_with_retry(client: OpenAI, messages: list, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
            return response
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
    return None

# COMMENT: I would use this in a production pipeline to handle transient network errors, API rate limits (HTTP 429), or temporary OpenAI server timeouts. Instead of crashing a 10,000-row batch halfway through, the script simply waits 2 seconds and tries the failed row again.