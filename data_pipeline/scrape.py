"""Step 1 - Scrape raw book listings from books.toscrape.com.

For each requested category we follow the sidebar link from the home page,
walk every paginated listing page (via the "next" button), and capture the
raw, *uncleaned* fields exactly as the site shows them:
title, price (e.g. "£51.77"), star_rating (e.g. "Three"), availability
(e.g. "In stock") and category.
"""

import time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

try:
    # Verify HTTPS against the operating system's certificate store instead of
    # certifi's bundle. Needed on machines where antivirus software (e.g. Avast
    # Web Shield) re-signs HTTPS traffic with its own locally-trusted root.
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

BASE_URL = "https://books.toscrape.com/"
CATEGORIES = ["Mystery", "Fantasy", "Historical Fiction"]  # 32 + 48 + 26 books
REQUEST_DELAY_SECONDS = 0.3  # be polite to the practice site
DATA_DIR = Path(__file__).parent / "data"
RAW_CSV = DATA_DIR / "raw_books.csv"

session = requests.Session()
session.headers["User-Agent"] = "zepto-capstone-scraper/1.0 (learning project)"


def get_soup(url: str) -> BeautifulSoup:
    """Download a page and parse it. Raises on any non-2xx status code."""
    response = session.get(url, timeout=20)
    response.raise_for_status()
    # Pass raw bytes so BeautifulSoup reads the page's own UTF-8 declaration;
    # response.text would guess ISO-8859-1 and turn "£" into "Â£".
    return BeautifulSoup(response.content, "html.parser")


def find_category_urls(names: list[str]) -> dict[str, str]:
    """Map each category name to its listing URL using the home page sidebar."""
    soup = get_soup(BASE_URL)
    links = soup.select("div.side_categories ul li ul li a")
    available = {a.get_text(strip=True): urljoin(BASE_URL, a["href"]) for a in links}
    missing = [n for n in names if n not in available]
    if missing:
        raise ValueError(f"Categories not found on site: {missing}")
    return {name: available[name] for name in names}


def parse_listing_page(soup: BeautifulSoup, category: str) -> list[dict]:
    """Extract the raw fields of every book card on one listing page."""
    books = []
    for card in soup.select("article.product_pod"):
        title_link = card.select_one("h3 a")
        rating_tag = card.select_one("p.star-rating")
        price_tag = card.select_one("p.price_color")
        avail_tag = card.select_one("p.availability")

        # The class list looks like ["star-rating", "Three"]; keep the word.
        rating_classes = rating_tag.get("class", []) if rating_tag else []
        star_rating = next((c for c in rating_classes if c != "star-rating"), None)

        books.append({
            # The visible <a> text is truncated ("A Light in the ..."), the
            # title attribute holds the full title.
            "title": title_link.get("title") if title_link else None,
            "price": price_tag.get_text(strip=True) if price_tag else None,
            "star_rating": star_rating,
            "availability": avail_tag.get_text(strip=True) if avail_tag else None,
            "category": category,
        })
    return books


def scrape_category(name: str, url: str) -> list[dict]:
    """Scrape all pages of one category by following the "next" link."""
    books, page = [], 1
    while url:
        soup = get_soup(url)
        page_books = parse_listing_page(soup, name)
        books.extend(page_books)
        print(f"  {name}: page {page} -> {len(page_books)} books")

        next_link = soup.select_one("li.next a")
        url = urljoin(url, next_link["href"]) if next_link else None
        page += 1
        time.sleep(REQUEST_DELAY_SECONDS)
    return books


def scrape(categories: list[str] = CATEGORIES) -> pd.DataFrame:
    """Scrape the given categories and save the raw rows to data/raw_books.csv."""
    print(f"Scraping {len(categories)} categories from {BASE_URL}")
    rows = []
    for name, url in find_category_urls(categories).items():
        rows.extend(scrape_category(name, url))

    raw_df = pd.DataFrame(rows)
    DATA_DIR.mkdir(exist_ok=True)
    raw_df.to_csv(RAW_CSV, index=False, encoding="utf-8")
    print(f"Scraped {len(raw_df)} books across {raw_df['category'].nunique()} "
          f"categories -> {RAW_CSV.relative_to(Path(__file__).parent)}")
    return raw_df


if __name__ == "__main__":
    scrape()
