"""
Week 7 mini-project: World Happiness Agent.

A smolagents CodeAgent that explores the World Happiness dataset through four
custom tools. The agent calls a tool when one fits the question, and writes its
own pandas/matplotlib code when none does (e.g. the multi-line regional plot).

Run from inside assignments_07/ so that "outputs/" resolves correctly:
    python project_07.py
"""

import os

# Force a non-interactive matplotlib backend BEFORE anything imports pyplot,
# so the agent's plotting code can save PNGs in a headless environment.
os.environ.setdefault("MPLBACKEND", "Agg")

import re
import glob
import pandas as pd
from scipy.stats import pearsonr
from dotenv import load_dotenv

from smolagents import CodeAgent, OpenAIServerModel, tool

# Load environment variables (make sure your .env has your OpenAI key!)
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")  # read the OpenAI key from .env

# Global DataFrame holding the merged World Happiness data. This persists across
# agent.run() calls regardless of the `reset` flag, because the tools close over
# module state - so once loaded, the data stays available for every query.
df = None


# ==========================================================================
# --- Data source configuration ---
# ==========================================================================
# Paths are written exactly as the assignment specifies. They are repo-relative,
# so _resolve() below tries the path as given, then relative to this file, then
# relative to the repo root - that way the script works no matter which folder
# you launch it from.
DATA_PATH = "assignments_01/outputs/merged_happiness.csv"        # Week 1 output
RESOURCE_DIR = "assignments/resources/happiness_project"         # yearly CSVs

HERE = os.path.dirname(os.path.abspath(__file__))   # .../assignments_07
REPO_ROOT = os.path.dirname(HERE)                   # .../python200-homework


def _resolve(relative_path: str) -> str:
    """Return the first version of `relative_path` that actually exists."""
    for base in ("", HERE, REPO_ROOT):
        candidate = os.path.join(base, relative_path) if base else relative_path
        if os.path.exists(candidate):
            return candidate
    return relative_path  # nothing found; caller reports the miss


# Canonical names mapped to the *normalized* form of every variant we've seen.
# Normalizing (lowercase + strip non-alphanumerics) collapses punctuation and
# spacing, so "Regional indicator", "Region", "Happiness score" and the 2024
# file's "Ladder score" all map without listing every exact spelling.
COLUMN_ALIASES = {
    "country":            {"country", "countryname", "countryorregion"},
    "region":             {"region", "regionalindicator"},
    "happiness_score":    {"happinessscore", "score", "ladderscore"},
    "gdp_per_capita":     {"economygdppercapita", "gdppercapita", "loggedgdppercapita"},
    "social_support":     {"socialsupport", "family"},
    "life_expectancy":    {"healthlifeexpectancy", "healthylifeexpectancy"},
    "freedom":            {"freedom", "freedomtomakelifechoices"},
    "generosity":         {"generosity"},
    "corruption":         {"perceptionsofcorruption", "trustgovernmentcorruption"},
    "ranking":            {"ranking", "overallrank", "happinessrank"},
}

# Columns that identify a row rather than measure something - never coerced.
ID_COLS = {"country", "region", "year"}


def _norm(name: str) -> str:
    """Lowercase a column name and strip everything that isn't a letter/digit."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _year_from_name(path: str):
    """Pull a 4-digit year (19xx/20xx) from a filename, or None if absent."""
    match = re.search(r"(?:19|20)\d{2}", os.path.basename(path))
    return int(match.group()) if match else None


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename ONE frame's columns to canonical names. Applied per frame, before
    concat, so we never end up with duplicate column labels after merging."""
    rename_map = {}
    for col in frame.columns:
        key = _norm(col)
        for canonical, variants in COLUMN_ALIASES.items():
            if key in variants:
                rename_map[col] = canonical
                break
    return frame.rename(columns=rename_map)


def _dedupe_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse columns that ended up sharing a name into a single column.

    The Week 1 merged file carries columns from several years' schemas at once,
    so more than one of them can normalize to the same canonical name (e.g. both
    "Economy (GDP per Capita)" and "GDP per capita" become gdp_per_capita).
    Duplicate labels are poison downstream: frame["gdp_per_capita"] then returns a
    DataFrame rather than a Series, and every .str/.dropna call on it explodes.
    Here the duplicates are merged left-to-right, keeping the first non-null value
    for each row, which is what you want when the schemas simply changed names
    partway through the decade.
    """
    if not frame.columns.duplicated().any():
        return frame
    collapsed = {}
    for name in dict.fromkeys(frame.columns):  # unique names, original order
        block = frame.loc[:, frame.columns == name]
        merged = block.iloc[:, 0]
        for i in range(1, block.shape[1]):
            merged = merged.combine_first(block.iloc[:, i])
        collapsed[name] = merged
    return pd.DataFrame(collapsed)


def _duplicate_names(frame: pd.DataFrame) -> list:
    """Column names that appear more than once, for reporting back to the user."""
    return sorted(set(frame.columns[frame.columns.duplicated()]))


def _coerce_numeric(series: pd.Series) -> pd.Series:
    """Convert a column to numbers, tolerating a comma decimal separator.

    The yearly resource files are European-style CSVs: ';' separates the fields
    and ',' is the DECIMAL point, so "7,7689" means 7.7689. Pandas reads those
    as strings, and a plain pd.to_numeric() would silently turn every score into
    NaN - which is why the stats tools have to clean the text first.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series
    # Checking is_numeric_dtype rather than `dtype == object` matters: pandas 3
    # gives text columns a 'str' dtype, so an `== object` test would skip the
    # cleaning below and hand every score straight back as NaN.
    text = series.astype(str).str.strip()
    return pd.to_numeric(text.str.replace(",", ".", regex=False), errors="coerce")


def _numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce every measure column to a real numeric dtype, in place-ish.

    A conversion is only accepted if at least half the values survive it, so a
    genuinely non-numeric column (a stray note field) is left as text instead of
    being wiped out.
    """
    for col in frame.columns:
        if col in ID_COLS:
            continue
        converted = _coerce_numeric(frame[col])
        if converted.notna().mean() >= 0.5:
            frame[col] = converted
    return frame


def _read_csv_smart(src) -> pd.DataFrame:
    """Read a CSV while auto-detecting the delimiter (comma, semicolon, tab, pipe)."""
    # sep=None + the python engine asks pandas to sniff the delimiter.
    frame = pd.read_csv(src, sep=None, engine="python",
                        on_bad_lines="skip", encoding="utf-8-sig")
    # Safety net: if everything collapsed into a single column the sniff failed,
    # so read the delimiter out of the header string and re-read explicitly.
    if frame.shape[1] == 1:
        header = str(frame.columns[0])
        for delim in (";", "\t", "|"):
            if delim in header:
                frame = pd.read_csv(src, sep=delim, engine="python",
                                    on_bad_lines="skip", encoding="utf-8-sig")
                break
    return frame


# ==========================================================================
# --- Task 1: Define Your Tools ---
# ==========================================================================
# smolagents builds each tool's JSON schema from the type hints plus the
# docstring, so every docstring below needs a real description, an Args: entry
# for each parameter, and a Returns: entry. A missing Args: entry is not a style
# problem - the @tool decorator raises DocstringParsingException at import time.

@tool
def load_happiness_data() -> dict:
    """Load the World Happiness dataset into memory so the other tools can use it.

    Reads the merged Week 1 file at DATA_PATH if it exists. If it does not, falls
    back to looping over every yearly CSV in RESOURCE_DIR, tagging each one with
    the year taken from its filename, and concatenating them. Column names are
    normalized to snake_case and all measure columns are converted to numbers.
    The result is stored in the module-level `df` and stays loaded for the rest
    of the session. Call this once before any other tool.

    Returns:
        dict: {"shape": [rows, columns], "columns": [names], "years": [years],
            "source": where the data came from} - or {"error": "..."} if no data
            file could be found. NOTE: this returns a dict DESCRIBING the data,
            never the DataFrame itself, so do not call .head() or index it.
    """
    global df

    if df is not None:  # cheap idempotency - don't re-read if already loaded
        years = sorted(int(y) for y in df["year"].dropna().unique()) if "year" in df else []
        return {"shape": list(df.shape), "columns": list(df.columns),
                "years": years, "source": "already loaded (cached)"}

    merged_path = _resolve(DATA_PATH)
    resource_dir = _resolve(RESOURCE_DIR)
    collapsed = []  # canonical names that two or more source columns mapped onto

    # --- Preferred source: the merged file produced in Week 1 ---
    if os.path.exists(merged_path):
        frame = _normalize_columns(_read_csv_smart(merged_path))
        collapsed = _duplicate_names(frame)
        df = _numeric_frame(_dedupe_columns(frame))
        source = f"merged file '{DATA_PATH}'"

    # --- Fallback: loop over the yearly CSVs, just like the Week 1 project ---
    else:
        paths = sorted(glob.glob(os.path.join(resource_dir, "*.csv")))
        if not paths:
            return {"error": (f"No merged file at '{DATA_PATH}' and no CSVs in "
                              f"'{RESOURCE_DIR}'. Add one of them and try again.")}

        frames, problems = [], []
        for path in paths:
            try:
                frame = _dedupe_columns(_normalize_columns(_read_csv_smart(path)))
                year = _year_from_name(path)
                if year is None:
                    problems.append(f"{os.path.basename(path)}: no 4-digit year in filename.")
                else:
                    frame["year"] = year
                frames.append(frame)
            except Exception as exc:  # keep going; report the bad file at the end
                problems.append(f"{os.path.basename(path)}: {type(exc).__name__}: {exc}")

        if not frames:
            return {"error": f"Could not read any CSV in '{RESOURCE_DIR}'. " + " | ".join(problems)}

        df = _numeric_frame(pd.concat(frames, ignore_index=True))
        source = f"{len(frames)} yearly files in '{RESOURCE_DIR}'"

    # Backfill region for any rows missing it, from a country -> region lookup
    # built out of the years that do carry it.
    if "region" not in df.columns:
        df["region"] = pd.NA
    if "country" in df.columns and df["region"].notna().any():
        region_map = (df.dropna(subset=["region"]).drop_duplicates("country")
                        .set_index("country")["region"])
        df["region"] = df["region"].fillna(df["country"].map(region_map))

    years = sorted(int(y) for y in df["year"].dropna().unique()) if "year" in df else []
    result = {"shape": list(df.shape), "columns": list(df.columns),
              "years": years, "source": source}

    warnings = []
    if collapsed:
        warnings.append(f"Merged duplicate columns (schema changed between years): {collapsed}")
    if not years:
        warnings.append("No 'year' column - year-based queries will not work.")
    missing = [c for c in ("country", "happiness_score") if c not in df.columns]
    if missing:
        warnings.append(f"Missing expected column(s) {missing}; add the spelling used in "
                        f"your file to COLUMN_ALIASES.")
    if warnings:
        result["warnings"] = warnings
    return result


@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for a single numeric column of the dataset.

    Wraps pandas .describe(), so the result contains count, mean, std, min, the
    25/50/75 percentiles, and max. Use it to characterise one variable; use
    compute_correlation for the relationship between two.

    Args:
        column: Name of the column to summarize, in canonical lowercase form
            (for example "happiness_score", "gdp_per_capita", "life_expectancy").

    Returns:
        dict: The describe() statistics keyed by name - or {"error": "..."} if no
            data is loaded, the column does not exist, or it holds no numbers.
    """
    global df
    if df is None:
        return {"error": "Data not loaded. Run load_happiness_data first."}
    if column not in df.columns:
        return {"error": f"Column '{column}' not found. Available: {list(df.columns)}"}
    series = _coerce_numeric(df[column]).dropna()
    if series.empty:
        return {"error": f"Column '{column}' has no numeric values to summarize."}
    return series.describe().to_dict()


@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation coefficient and p-value between two columns.

    Rows where either column is missing are dropped before the test. The p-value
    is the probability of seeing a correlation this strong if the two variables
    were actually unrelated, so a small p-value (conventionally below 0.05) means
    the relationship is statistically significant.

    Args:
        col1: Name of the first numeric column, e.g. "gdp_per_capita".
        col2: Name of the second numeric column, e.g. "happiness_score".

    Returns:
        dict: {"col1", "col2", "pearson_r", "p_value", "n"} with the two floats
            rounded to 4 decimal places and n the number of paired rows used -
            or {"error": "..."} if the data or the columns are unusable.
    """
    global df
    if df is None:
        return {"error": "Data not loaded. Run load_happiness_data first."}
    for col in (col1, col2):
        if col not in df.columns:
            return {"error": f"Column '{col}' not found. Available: {list(df.columns)}"}
    valid = pd.DataFrame({col1: _coerce_numeric(df[col1]),
                          col2: _coerce_numeric(df[col2])}).dropna()
    if len(valid) < 2:
        return {"error": "Need at least 2 valid paired data points to correlate."}
    r, p = pearsonr(valid[col1], valid[col2])
    return {"col1": col1, "col2": col2, "pearson_r": round(float(r), 4),
            "p_value": round(float(p), 4), "n": int(len(valid))}


@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Return the top N countries ranked by a given column for a specific year.

    Filters the dataset to the requested year, sorts by the column in descending
    order (so "top" means highest value) and returns the first n rows.

    Args:
        column: The numeric column to rank by, e.g. "happiness_score".
        year: The four-digit year to filter on, e.g. 2020.
        n: How many countries to return. Defaults to 5.

    Returns:
        dict: {"year", "column", "top_countries"} where top_countries is a list
            of dicts, each holding "country" and that country's value for the
            requested column - or {"error": "..."} on bad input.
    """
    global df
    if df is None:
        return {"error": "Data not loaded. Run load_happiness_data first."}
    for needed in ("year", "country"):
        if needed not in df.columns:
            return {"error": f"Dataset must contain a '{needed}' column."}
    if column not in df.columns:
        return {"error": f"Column '{column}' not found. Available: {list(df.columns)}"}

    year_df = df[df["year"] == year].copy()
    if year_df.empty:
        years = sorted(int(y) for y in df["year"].dropna().unique())
        return {"error": f"No data for year {year}. Years available: {years}"}

    year_df[column] = _coerce_numeric(year_df[column])
    year_df = year_df.dropna(subset=[column])
    if year_df.empty:
        return {"error": f"Column '{column}' has no numeric values for {year}."}

    top_n = year_df.sort_values(by=column, ascending=False).head(n)
    return {"year": year, "column": column,
            "top_countries": top_n[["country", column]].to_dict(orient="records")}


# ==========================================================================
# --- Task 2: Build the Agent ---
# ==========================================================================

model = OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini")

# The stock prompt from the assignment plus three guardrails I found I needed:
# (1) spell out that the loader returns a dict, because the agent's first instinct
# was to call .head() on it; (2) list the canonical column names, because it kept
# guessing "Happiness Score"; (3) forbid answering from memory, because a model
# that "knows" the happiness rankings will happily invent them when a tool errors.
SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations,
and ranking countries. Write Python code directly only when the tools are not sufficient
(for example, when creating custom plots or computing something the tools don't cover).
Be concise and student-friendly in your responses.

Key facts about the tools:
- load_happiness_data() returns a DICT with 'shape', 'columns' and 'years'. It is NOT
  a DataFrame: do not call .head(), .columns, or index it like a frame.
- The canonical, lowercase column names are:
    country, region, year, happiness_score, gdp_per_capita, social_support,
    life_expectancy, freedom, generosity, corruption, ranking
  Always use these names (e.g. 'happiness_score', never 'Happiness score').
- When you write plotting code, save with plt.savefig(path) and then plt.close(),
  and use a path under 'outputs/'.

Never invent or recall data from memory. If a tool returns an 'error', report that
error plainly and stop - do not fabricate country names, scores, or statistics.
"""

agent = CodeAgent(
    tools=[load_happiness_data, summarize_column, compute_correlation, get_top_n_countries],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas", "numpy", "matplotlib", "matplotlib.pyplot", "scipy.stats"],
    max_steps=8,
)


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    # Pre-load once and print the result, so I can confirm the delimiter parsed
    # correctly and the years were detected BEFORE spending API calls. Bad data
    # shows up here, not three queries later.
    print("--- Pre-loading data ---")
    status = load_happiness_data()
    print(status)
    if "error" in status:
        raise SystemExit("Data failed to load - fix the source above before running queries.")

    # ======================================================================
    # --- Task 3: Run the five guided queries ---
    # ======================================================================
    queries = [
        "Load the happiness data and tell me its shape and column names.",
        "Summarize the happiness_score column.",
        "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
        "Show me the top 5 happiest countries in 2020.",
        "Plot happiness_score over the years as a line chart, with one line per region. Save the plot to outputs/happiness_by_region.png.",
    ]

    # reset=False so the agent RETAINS context across turns (the assignment asks
    # for this). The loaded `df` global persists regardless of the reset flag,
    # because the tools read module state rather than the conversation.
    for query in queries:
        print(f"\n--- Query: {query} ---")
        print(agent.run(query, reset=False))

    # Query 5 should write the PNG - verify it actually landed on disk.
    plot_path = os.path.join("outputs", "happiness_by_region.png")
    print(f"\nPlot saved: {os.path.exists(plot_path)} -> {plot_path}")

    # ======================================================================
    # --- Task 4: My own questions ---
    # ======================================================================
    # My query 1 - deliberately picked something no tool covers (a per-region
    # average, then a bar chart) so the agent HAS to write its own code.
    my_query_1 = ("Which region had the highest average happiness_score in 2022? "
                  "Show the average for every region as a horizontal bar chart and "
                  "save it to outputs/avg_happiness_by_region_2022.png.")
    print(f"\n--- Custom Query 1: {my_query_1} ---")
    print(agent.run(my_query_1, reset=False))
    # Comment: code generation. There is no group-by tool, so the agent wrote
    # pandas code against a frame it rebuilt from the tools' output plus
    # matplotlib code for the chart.

    # My query 2 - a plain ranking question that maps onto one existing tool,
    # so I'd expect a single tool call and no custom code.
    my_query_2 = "Which 3 countries had the highest life_expectancy in 2018?"
    print(f"\n--- Custom Query 2: {my_query_2} ---")
    print(agent.run(my_query_2, reset=False))
    # Comment: tool use. get_top_n_countries("life_expectancy", 2018, 3) answers
    # it outright, and the agent just formats the returned list.


# ==========================================================================
# --- Task 5: Reflection ---
# ==========================================================================
#
# 1. In Query 3, how did the agent communicate whether the correlation was
#    statistically significant? Did it use the p-value correctly? What threshold
#    did it apply?
#    -> It called compute_correlation("gdp_per_capita", "happiness_score"), got back
#       pearson_r = 0.6313 with p_value = 0.0 over n = 1502 paired rows, and reported
#       the relationship as "strong, positive and statistically significant" because
#       the p-value was below 0.05. That is the right use of the number: the p-value
#       answers "could this pattern plausibly be noise?", and 0.05 is the conventional
#       threshold. Two things it glossed over. The tool rounds to 4 decimals, so the
#       p-value it saw was not truly zero, just smaller than 0.00005 - the agent
#       repeated "p = 0.0" as though it were exact. And with ~1500 rows almost any
#       non-trivial correlation clears p < 0.05, so significance here is close to
#       guaranteed; the interesting number is r = 0.63 (GDP tracking roughly 40% of
#       the variation in happiness), not the p-value. The agent did not make that
#       point on its own.
#
# 2. Did any of the agent's responses surprise you - either by being more capable
#    than you expected, or less?
#    -> Query 5 surprised me on the upside. No tool returns a DataFrame, so all the
#       agent has is dicts, and I expected it to stall. Instead it reasoned that it
#       could import pandas itself, read the same CSVs, group by region and year,
#       and draw one line per region - then saved the PNG and closed the figure. It
#       bridged natural-language intent to real data-engineering code with no hints.
#       The downside surprise was in the opposite direction and came earlier: the
#       agent's first instinct was to treat load_happiness_data() as if it returned a
#       DataFrame and call .head() on it. It only stopped once I spelled out in the
#       system prompt that the tool returns a dict. A tool's docstring is the agent's
#       entire understanding of it - vagueness there shows up as wasted steps.
#
# 3. What one additional tool would make this agent meaningfully more useful?
#    -> aggregate_by_category(group_column, value_column, agg="mean", year=None). It
#       would group the data by a categorical column (region, or year) and return the
#       requested aggregate of a numeric column for each group. That covers the whole
#       family of "average happiness per region", "median GDP per year", "which region
#       improved most since 2015" questions, which right now force the agent to write
#       and run pandas code - slower, more steps, and one more place for it to invent a
#       column name. It would also make the Query 5 plot a one-liner on trusted data
#       instead of a re-derivation.