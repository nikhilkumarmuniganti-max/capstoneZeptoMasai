"""Step 3 - Load the cleaned data into a normalized two-table SQLite schema.

    categories(category_id PK, category_name UNIQUE)
            1 ──< many
    books(book_id PK, title, price_gbp, price_inr, rating, in_stock,
          category_id FK -> categories.category_id)

Category names are stored once in `categories`; each book row references its
category through the integer foreign key instead of repeating the name.
The database file is deleted and rebuilt from scratch on every run, so this
script is the exact recipe that regenerates data/zepto_books.db.
"""

import sqlite3
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
DB_PATH = DATA_DIR / "zepto_books.db"

SCHEMA = """
CREATE TABLE categories (
    category_id   INTEGER PRIMARY KEY,
    category_name TEXT NOT NULL UNIQUE
);

CREATE TABLE books (
    book_id     INTEGER PRIMARY KEY,
    title       TEXT    NOT NULL,
    price_gbp   REAL    NOT NULL,
    price_inr   REAL    NOT NULL,
    rating      INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    in_stock    INTEGER NOT NULL CHECK (in_stock IN (0, 1)),
    category_id INTEGER NOT NULL REFERENCES categories(category_id)
);
"""


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")  # SQLite enforces FKs only when asked
    return conn


def split_tables(clean_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize the flat cleaned frame into categories and books frames."""
    categories_df = pd.DataFrame({
        "category_name": sorted(clean_df["category"].unique())
    })
    categories_df.insert(0, "category_id", range(1, len(categories_df) + 1))

    books_df = clean_df.merge(categories_df, left_on="category",
                              right_on="category_name", how="left")
    books_df.insert(0, "book_id", range(1, len(books_df) + 1))
    books_df["in_stock"] = books_df["in_stock"].astype(int)  # SQLite has no bool
    books_df = books_df[["book_id", "title", "price_gbp", "price_inr",
                         "rating", "in_stock", "category_id"]]
    return categories_df, books_df


def load(clean_df: pd.DataFrame, db_path: Path = DB_PATH):
    """(Re)create the database and insert both tables.

    Returns the in-memory (categories_df, books_df) frames that were inserted,
    so the pandas-only merge can be compared against the SQL JOIN later.
    """
    categories_df, books_df = split_tables(clean_df)

    DATA_DIR.mkdir(exist_ok=True)
    db_path.unlink(missing_ok=True)
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
        conn.executemany(
            "INSERT INTO categories (category_id, category_name) VALUES (?, ?)",
            categories_df.itertuples(index=False, name=None),
        )
        conn.executemany(
            "INSERT INTO books (book_id, title, price_gbp, price_inr, rating, "
            "in_stock, category_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
            books_df.itertuples(index=False, name=None),
        )
        n_cat = conn.execute("SELECT COUNT(*) FROM categories").fetchone()[0]
        n_books = conn.execute("SELECT COUNT(*) FROM books").fetchone()[0]
    conn.close()

    print(f"Loaded {n_cat} categories and {n_books} books "
          f"-> {db_path.relative_to(Path(__file__).parent)}")
    return categories_df, books_df


if __name__ == "__main__":
    from clean import CLEAN_CSV
    load(pd.read_csv(CLEAN_CSV))
