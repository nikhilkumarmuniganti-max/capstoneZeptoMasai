"""Step 2 - Clean the raw scraped fields into proper types and convert currency.

Output columns:
    title (str), category (str), price_gbp (float), rating (int 1-5),
    in_stock (bool), price_inr (float)

Handling of rows that fail to parse (the pipeline never crashes on messy rows):
    * price / star_rating (numeric fields)  -> median imputation, using the
      median of the rows that DID parse. One bad value should not cost us an
      otherwise valid book, and the median is robust to price outliers.
    * availability (boolean field)          -> drop the row. A yes/no field has
      no meaningful median, and guessing stock status would publish wrong
      availability data to analysts.
    * title / category missing              -> drop the row. These identify the
      book and its category foreign key; they cannot be imputed.
"""

import re
from pathlib import Path

import pandas as pd

# Project-defined fixed baseline rate (not a live or historical market rate).
GBP_TO_INR = 105.50

RATING_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
DATA_DIR = Path(__file__).parent / "data"
RAW_CSV = DATA_DIR / "raw_books.csv"
CLEAN_CSV = DATA_DIR / "clean_books.csv"


def parse_price(value) -> float | None:
    """'£51.77' -> 51.77. Returns None if no number can be found."""
    if pd.isna(value):
        return None
    match = re.search(r"\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group()) if match else None


def parse_rating(value) -> int | None:
    """'Three' -> 3. Returns None for anything outside One..Five."""
    if pd.isna(value):
        return None
    return RATING_WORDS.get(str(value).strip().lower())


def parse_in_stock(value) -> bool | None:
    """'In stock' / 'In stock (22 available)' -> True, 'Out of stock' -> False."""
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text.startswith("out of stock"):
        return False
    if text.startswith("in stock"):
        return True
    return None


def clean(raw_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Return a typed, currency-converted copy of the raw scraped data."""
    df = raw_df.copy()
    log = print if verbose else (lambda *a, **k: None)

    df["title"] = df["title"].str.strip()
    df["category"] = df["category"].str.strip()
    df["price_gbp"] = df["price"].map(parse_price)
    df["rating"] = df["star_rating"].map(parse_rating)
    df["in_stock"] = df["availability"].map(parse_in_stock)

    # Drop rows whose identity or boolean fields could not be parsed.
    unusable = df["title"].isna() | df["category"].isna() | df["in_stock"].isna()
    if unusable.any():
        log(f"Dropping {unusable.sum()} row(s) with missing title/category "
            f"or unrecognised availability text")
    df = df[~unusable].copy()

    # Median-impute numeric fields that failed to parse.
    for col in ["price_gbp", "rating"]:
        n_missing = df[col].isna().sum()
        if n_missing:
            median = df[col].median()
            if col == "rating":
                median = round(median)
            log(f"Imputing {n_missing} unparseable '{col}' value(s) with median {median}")
            df[col] = df[col].fillna(median)

    df["price_gbp"] = df["price_gbp"].astype(float)
    df["rating"] = df["rating"].astype(int)
    df["in_stock"] = df["in_stock"].astype(bool)
    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR).round(2)

    return df[["title", "category", "price_gbp", "rating", "in_stock", "price_inr"]] \
        .reset_index(drop=True)


def messy_rows_demo() -> pd.DataFrame:
    """Show that clean() survives malformed rows instead of crashing."""
    messy = pd.DataFrame([
        {"title": "Good Book", "price": "£10.00", "star_rating": "Five",
         "availability": "In stock", "category": "Demo"},
        {"title": "Bad Price", "price": "N/A", "star_rating": "Two",
         "availability": "In stock", "category": "Demo"},
        {"title": "Bad Rating", "price": "£30.00", "star_rating": "Seven",
         "availability": "Out of stock", "category": "Demo"},
        {"title": "Bad Availability", "price": "£20.00", "star_rating": "One",
         "availability": "Ask in store", "category": "Demo"},
        {"title": None, "price": "£15.00", "star_rating": "Three",
         "availability": "In stock", "category": "Demo"},
    ])
    print("Messy input:")
    print(messy.to_string(index=False))
    print()
    cleaned = clean(messy)
    print("\nCleaned output:")
    print(cleaned.to_string(index=False))
    return cleaned


def run() -> pd.DataFrame:
    """Clean data/raw_books.csv and save data/clean_books.csv."""
    raw_df = pd.read_csv(RAW_CSV)
    clean_df = clean(raw_df)
    clean_df.to_csv(CLEAN_CSV, index=False, encoding="utf-8")
    print(f"Cleaned {len(clean_df)} of {len(raw_df)} rows "
          f"(1 GBP = {GBP_TO_INR} INR) -> {CLEAN_CSV.relative_to(Path(__file__).parent)}")
    print(clean_df.dtypes.to_string())
    return clean_df


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")  # show "£" correctly on Windows consoles
    run()
    print("\n--- Robustness check on deliberately messy rows ---")
    messy_rows_demo()
