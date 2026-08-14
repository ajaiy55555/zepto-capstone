"""
Builds a normalized SQLite database from books_clean.csv:
  categories(category_id PK, category_name)
  books(book_id PK, title, price_gbp, price_inr, rating, in_stock, category_id FK)

Usage:
    python build_db.py
"""

import os
import sqlite3
import pandas as pd

DB_PATH = "books.db"


def main():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT UNIQUE NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price_gbp REAL,
            price_inr REAL,
            rating INTEGER,
            in_stock INTEGER,
            category_id INTEGER,
            FOREIGN KEY (category_id) REFERENCES categories(category_id)
        )
    """)
    conn.commit()

    df = pd.read_csv("books_clean.csv")

    for name in sorted(df["category"].unique()):
        cursor.execute("INSERT INTO categories (category_name) VALUES (?)", (name,))
    conn.commit()

    cursor.execute("SELECT category_id, category_name FROM categories")
    category_lookup = {name: cid for cid, name in cursor.fetchall()}

    for _, row in df.iterrows():
        cursor.execute(
            """
            INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                row["title"],
                float(row["price_gbp"]),
                float(row["price_inr"]),
                int(row["rating"]),
                1 if bool(row["in_stock"]) else 0,
                category_lookup[row["category"]],
            ),
        )
    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM categories")
    n_categories = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM books")
    n_books = cursor.fetchone()[0]
    print(f"Loaded {n_categories} categories and {n_books} books into {DB_PATH}")

    conn.close()


if __name__ == "__main__":
    main()
