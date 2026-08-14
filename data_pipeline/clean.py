"""
Cleans raw_books.csv into properly typed columns and adds an INR price
column using the project's fixed conversion rate.

Usage:
    python clean.py
"""

import pandas as pd

GBP_TO_INR_RATE = 105.50  # fixed project-defined rate, not a live market rate

RATING_WORD_TO_NUMBER = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


def clean_price(price_raw: str):
    try:
        return float(str(price_raw).replace("£", "").strip())
    except (ValueError, TypeError):
        return None


def clean_rating(rating_text: str):
    return RATING_WORD_TO_NUMBER.get(str(rating_text).strip(), None)


def clean_in_stock(availability_raw: str) -> bool:
    return "in stock" in str(availability_raw).strip().lower()


def main():
    df = pd.read_csv("raw_books.csv")

    df["price_gbp"] = df["price_raw"].apply(clean_price)
    df["rating"] = df["star_rating_text"].apply(clean_rating)
    df["in_stock"] = df["availability_raw"].apply(clean_in_stock)

    # Numeric fields: impute missing values with the column median rather
    # than drop the row, since a missing price/rating doesn't invalidate
    # the rest of that book's data.
    df["price_gbp"] = df["price_gbp"].fillna(df["price_gbp"].median())
    df["rating"] = df["rating"].fillna(df["rating"].median()).astype(int)

    # Identifying fields: a row with no title or category isn't usable,
    # so such rows are dropped instead of imputed.
    df = df.dropna(subset=["title", "category"])

    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR_RATE).round(2)

    df_clean = df[["title", "category", "price_gbp", "price_inr", "rating", "in_stock"]]
    df_clean.to_csv("books_clean.csv", index=False)

    print(f"Cleaned {len(df_clean)} rows -> books_clean.csv")
    print(df_clean.head())


if __name__ == "__main__":
    main()
