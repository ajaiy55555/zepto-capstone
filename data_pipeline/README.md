# Module 1 — Data Pipeline

Scrapes book data from [books.toscrape.com](https://books.toscrape.com), cleans it,
converts prices to INR, and loads it into a normalized SQLite database that's then
queried with both SQL and pandas.

## Install

```bash
pip install -r requirements.txt
```

## Run (in order)

```bash
python scrape.py     # scrapes books.toscrape.com -> raw_books.csv
python clean.py       # cleans raw_books.csv -> books_clean.csv
python build_db.py    # builds books.db from books_clean.csv
python queries.py     # runs required SQL queries -> query_results.txt
```

Each script depends on the output of the one before it. `scrape.py` requires an
internet connection.

## Files

| File | Purpose |
|---|---|
| `scrape.py` | Scrapes 3 categories (Travel, Mystery, Historical Fiction) using `requests` + `BeautifulSoup`, handling pagination. Outputs `raw_books.csv`. |
| `clean.py` | Converts raw text fields to proper types (`price_gbp` float, `rating` int, `in_stock` bool) and adds `price_inr`. Outputs `books_clean.csv`. |
| `build_db.py` | Creates the normalized `categories`/`books` SQLite schema and loads the cleaned data. Outputs `books.db`. |
| `queries.py` | Runs 5 SQL queries covering SELECT/WHERE, ORDER BY, LIMIT, DISTINCT, IN/BETWEEN, and a JOIN. Verifies `pd.read_sql` and `pd.merge` produce equivalent results for the join. Outputs `query_results.txt`. |

## Design decisions

- **Currency conversion**: fixed rate of **1 GBP = 105.50 INR**, applied as
  `price_inr = price_gbp * 105.50`. This is the project-defined constant, not a
  live market rate — no external API is used.
- **Handling parse failures**: numeric fields (`price_gbp`, `rating`) are imputed
  with the column median if a value fails to parse, since that doesn't invalidate
  the rest of the row. Rows missing `title` or `category` are dropped, since those
  fields make the row unusable.
- **Normalization**: `categories` and `books` are separate tables linked by
  `category_id` (PK/FK), avoiding repeated category text on every book row.
- **Top-N-per-category query**: SQLite has no built-in "top N per group" clause,
  so the join query uses a correlated subquery (count same-category books with a
  strictly higher rating; keep the row if fewer than 3 do). The pandas equivalent
  uses `groupby("category_name")["rating"].rank(...)` to reproduce the same result
  without SQL.
