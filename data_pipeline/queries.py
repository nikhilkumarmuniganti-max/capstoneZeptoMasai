"""Step 4 - Query the database with SQL, then read results back with pandas.

1. Runs six SQL queries with sqlite3 that together cover
   SELECT/WHERE, ORDER BY, LIMIT, DISTINCT, IN, BETWEEN, JOIN and GROUP BY.
2. Reads two of those results into DataFrames with pd.read_sql.
3. Rebuilds the JOIN query's result with pd.merge on the in-memory
   DataFrames (no SQL) and checks both approaches give identical output.

Every query string and its output is printed and also saved to
outputs/query_results.md.
"""

from pathlib import Path

import pandas as pd

from database import DB_PATH, connect

OUTPUT_MD = Path(__file__).parent / "outputs" / "query_results.md"

QUERIES = [
    {
        "name": "Q1 - SELECT / WHERE",
        "question": "Which five-star books are in stock for under £20?",
        "sql": """
SELECT title, price_gbp, price_inr
FROM books
WHERE rating = 5
  AND in_stock = 1
  AND price_gbp < 20;""",
    },
    {
        "name": "Q2 - ORDER BY + LIMIT",
        "question": "What are the 10 most expensive books?",
        "sql": """
SELECT title, price_gbp, price_inr, rating
FROM books
ORDER BY price_gbp DESC
LIMIT 10;""",
    },
    {
        "name": "Q3 - DISTINCT",
        "question": "Which distinct star ratings appear in the catalogue?",
        "sql": """
SELECT DISTINCT rating
FROM books
ORDER BY rating DESC;""",
    },
    {
        "name": "Q4 - BETWEEN + IN",
        "question": "Which 4- or 5-star books cost between INR 2,000 and 3,000?",
        "sql": """
SELECT title, rating, price_inr
FROM books
WHERE price_inr BETWEEN 2000 AND 3000
  AND rating IN (4, 5)
ORDER BY price_inr;""",
    },
    {
        "name": "Q5 - JOIN (top 5 highest-rated books per category)",
        "question": "What are the 5 highest-rated books in each category? "
                    "(ties broken by higher price, then title A-Z)",
        "sql": """
SELECT category_name, rank_in_category, title, rating, price_gbp
FROM (
    SELECT c.category_name,
           b.title,
           b.rating,
           b.price_gbp,
           ROW_NUMBER() OVER (
               PARTITION BY c.category_id
               ORDER BY b.rating DESC, b.price_gbp DESC, b.title ASC
           ) AS rank_in_category
    FROM books AS b
    JOIN categories AS c ON b.category_id = c.category_id
)
WHERE rank_in_category <= 5
ORDER BY category_name, rank_in_category;""",
    },
    {
        "name": "Q6 - JOIN + GROUP BY (category summary)",
        "question": "How many books does each category have, and what are "
                    "their average price and rating?",
        "sql": """
SELECT c.category_name,
       COUNT(*)                    AS n_books,
       ROUND(AVG(b.price_gbp), 2)  AS avg_price_gbp,
       ROUND(AVG(b.price_inr), 2)  AS avg_price_inr,
       ROUND(AVG(b.rating), 2)     AS avg_rating
FROM books AS b
JOIN categories AS c ON b.category_id = c.category_id
GROUP BY c.category_name
ORDER BY n_books DESC;""",
    },
]


def format_rows(columns: list[str], rows: list[tuple]) -> str:
    """Render sqlite3 cursor rows as a plain fixed-width text table."""
    cells = [[str(v) for v in row] for row in rows]
    widths = [max([len(c)] + [len(r[i]) for r in cells]) for i, c in enumerate(columns)]
    line = lambda values: "  ".join(v.ljust(w) for v, w in zip(values, widths))
    out = [line(columns), line(["-" * w for w in widths])]
    out += [line(r) for r in cells]
    out.append(f"({len(rows)} row{'s' if len(rows) != 1 else ''})")
    return "\n".join(out)


def run_sql_queries(conn, report: list[str]) -> None:
    """Execute every query with a sqlite3 cursor and log query + output."""
    report.append("## Part 1 - SQL queries (executed with `sqlite3`)\n")
    for q in QUERIES:
        cursor = conn.execute(q["sql"])
        columns = [d[0] for d in cursor.description]
        table = format_rows(columns, cursor.fetchall())

        print(f"\n=== {q['name']} ===\n{q['question']}{q['sql']}\n\n{table}")
        report.append(f"### {q['name']}\n\n*{q['question']}*\n\n"
                      f"```sql{q['sql']}\n```\n\n```text\n{table}\n```\n")


def pandas_join(books_df: pd.DataFrame, categories_df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce Q5 with pd.merge on in-memory DataFrames - no SQL involved."""
    merged = pd.merge(books_df, categories_df, on="category_id", how="inner")
    merged = merged.sort_values(
        ["category_name", "rating", "price_gbp", "title"],
        ascending=[True, False, False, True],
    )
    merged["rank_in_category"] = merged.groupby("category_name").cumcount() + 1
    top5 = merged[merged["rank_in_category"] <= 5]
    return top5[["category_name", "rank_in_category", "title", "rating", "price_gbp"]] \
        .reset_index(drop=True)


def compare_read_sql_and_merge(conn, books_df, categories_df, report: list[str]) -> bool:
    """pd.read_sql for Q5 and Q6, then check Q5 against the pd.merge version."""
    report.append("## Part 2 - Reading results back with `pd.read_sql`\n")
    read_back = {}
    for q in (QUERIES[4], QUERIES[5]):
        read_back[q["name"]] = df = pd.read_sql(q["sql"], conn)
        print(f"\n=== pd.read_sql: {q['name']} ===\n{df.to_string(index=False)}")
        report.append(f"### `pd.read_sql` -> {q['name']}\n\n"
                      f"```text\n{df.to_string(index=False)}\n```\n")

    sql_join = read_back[QUERIES[4]["name"]]
    merge_join = pandas_join(books_df, categories_df)

    def display_view(df):  # shorter titles so the two tables fit side by side
        view = df[["category_name", "title", "rating", "price_gbp"]].copy()
        view["title"] = view["title"].str.slice(0, 32)
        return view

    side_by_side_text = pd.concat(
        {"pd.read_sql (SQL JOIN)": display_view(sql_join),
         "pd.merge (pandas only)": display_view(merge_join)},
        axis=1,
    ).to_string()

    try:
        pd.testing.assert_frame_equal(sql_join, merge_join, check_dtype=False)
        match, verdict = True, "MATCH - both approaches return identical rows, in the same order."
    except AssertionError as err:
        match, verdict = False, f"MISMATCH - {err}"

    print(f"\n=== Q5 side by side: pd.read_sql vs pd.merge ===\n{side_by_side_text}\n\n{verdict}")
    report.append(
        "## Part 3 - JOIN reproduced with `pd.merge` (no SQL)\n\n"
        "`pandas_join()` merges the in-memory `books_df` and `categories_df` on "
        "`category_id`, sorts with the same keys as the SQL window function "
        "(rating DESC, price_gbp DESC, title ASC), numbers rows within each "
        "category with `groupby().cumcount()` and keeps the top 5.\n\n"
        f"```text\n{side_by_side_text}\n```\n\n"
        f"**Equality check** (`pd.testing.assert_frame_equal` on all columns): **{verdict}**\n"
    )
    return match


def run(books_df: pd.DataFrame, categories_df: pd.DataFrame) -> bool:
    report = [
        "# Data pipeline - query results\n",
        "_Generated by `python run_pipeline.py` (see `queries.py`). "
        f"Database: `data/{DB_PATH.name}`._\n",
    ]
    with connect() as conn:
        run_sql_queries(conn, report)
        match = compare_read_sql_and_merge(conn, books_df, categories_df, report)
    conn.close()

    OUTPUT_MD.parent.mkdir(exist_ok=True)
    OUTPUT_MD.write_text("\n".join(report), encoding="utf-8")
    print(f"\nSaved all queries and outputs -> {OUTPUT_MD.relative_to(Path(__file__).parent)}")
    return match


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    with connect() as c:
        books = pd.read_sql("SELECT * FROM books", c)
        categories = pd.read_sql("SELECT * FROM categories", c)
    c.close()
    run(books, categories)
