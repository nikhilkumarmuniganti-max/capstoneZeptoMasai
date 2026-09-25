"""Run the whole data pipeline end to end:

    scrape -> clean + convert (GBP -> INR) -> load into SQLite -> query

Usage (from the data_pipeline folder):
    python run_pipeline.py                # scrape live data first
    python run_pipeline.py --skip-scrape  # reuse data/raw_books.csv (offline)
"""

import argparse
import sys

import pandas as pd

import clean
import database
import queries
import scrape


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # show "£" correctly on Windows consoles
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-scrape", action="store_true",
                        help="reuse the committed data/raw_books.csv instead of scraping")
    args = parser.parse_args()

    print("\n##### Step 1: scrape #####")
    if args.skip_scrape:
        raw_df = pd.read_csv(scrape.RAW_CSV)
        print(f"Skipped scraping; loaded {len(raw_df)} rows from {scrape.RAW_CSV.name}")
    else:
        raw_df = scrape.scrape()

    print("\n##### Step 2: clean + convert #####")
    clean_df = clean.clean(raw_df)
    clean_df.to_csv(clean.CLEAN_CSV, index=False, encoding="utf-8")
    print(f"{len(clean_df)} clean rows, 1 GBP = {clean.GBP_TO_INR} INR "
          f"-> {clean.CLEAN_CSV.name}")
    print(clean_df.dtypes.to_string())
    print("\nRobustness check on deliberately messy rows:")
    clean.messy_rows_demo()

    print("\n##### Step 3: load into SQLite #####")
    categories_df, books_df = database.load(clean_df)

    print("\n##### Step 4: SQL queries + pandas comparison #####")
    match = queries.run(books_df, categories_df)

    print("\nPipeline finished." if match else "\nPipeline finished WITH A JOIN MISMATCH.")
    sys.exit(0 if match else 1)


if __name__ == "__main__":
    main()
