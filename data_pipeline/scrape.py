"""
Scrapes book listings from books.toscrape.com across three categories
(Travel, Mystery, Historical Fiction) and writes the raw results to
raw_books.csv for downstream cleaning.

Usage:
    python scrape.py
"""

import time
import csv
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/catalogue/category/books/"
CATEGORIES = {
    "Travel": "travel_2",
    "Mystery": "mystery_3",
    "Historical Fiction": "historical-fiction_4",
}

HEADERS = {"User-Agent": "Mozilla/5.0 (educational scraping exercise)"}

WORD_TO_NUMBER = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


def get_soup(url: str) -> BeautifulSoup:
    response = requests.get(url, headers=HEADERS, timeout=10)
    response.raise_for_status()
    response.encoding = "utf-8"  # site's declared charset; avoids mojibake on £ and apostrophes
    return BeautifulSoup(response.text, "html.parser")


def scrape_category(category_name: str, slug: str) -> list[dict]:
    """Scrapes every book in a category, following pagination."""
    books = []
    page_url = f"{BASE_URL}{slug}/index.html"

    while page_url:
        soup = get_soup(page_url)

        for article in soup.select("article.product_pod"):
            title = article.h3.a["title"]
            price_text = article.select_one("p.price_color").text
            rating_word = article.p["class"][1]  # e.g. class="star-rating Three"
            availability_text = article.select_one("p.instock.availability").text.strip()

            books.append({
                "title": title,
                "price_raw": price_text,
                "star_rating_text": rating_word,
                "availability_raw": availability_text,
                "category": category_name,
            })

        next_link = soup.select_one("li.next a")
        page_url = page_url.rsplit("/", 1)[0] + "/" + next_link["href"] if next_link else None
        time.sleep(0.5)

    return books


def main():
    all_books = []
    for category_name, slug in CATEGORIES.items():
        all_books.extend(scrape_category(category_name, slug))

    print(f"Scraped {len(all_books)} books across {len(CATEGORIES)} categories.")

    fieldnames = ["title", "price_raw", "star_rating_text", "availability_raw", "category"]
    with open("raw_books.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_books)

    print("Saved raw_books.csv")


if __name__ == "__main__":
    main()
