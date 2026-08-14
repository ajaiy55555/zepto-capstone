"""
Runs the required SQL queries against books.db and verifies that
pd.read_sql and pd.merge produce equivalent results for the join query.

Usage:
    python queries.py
"""

import sqlite3
import pandas as pd

DB_PATH = "books.db"


def run_and_log(cursor, log_lines, title, sql):
    print(f"\n=== {title} ===")
    print(sql.strip())
    cursor.execute(sql)
    rows = cursor.fetchall()
    col_names = [d[0] for d in cursor.description]
    print(col_names)
    for row in rows:
        print(row)

    log_lines.append(f"\n=== {title} ===\n{sql.strip()}\n\n{col_names}\n")
    log_lines.extend(f"{row}\n" for row in rows)

    return rows, col_names


def main():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    log_lines = []

    run_and_log(cursor, log_lines, "Query 1: SELECT + WHERE (books under £20)", """
        SELECT title, price_gbp FROM books WHERE price_gbp < 20
    """)

    run_and_log(cursor, log_lines, "Query 2: ORDER BY + LIMIT (5 most expensive books)", """
        SELECT title, price_gbp FROM books ORDER BY price_gbp DESC LIMIT 5
    """)

    run_and_log(cursor, log_lines, "Query 3: DISTINCT category names", """
        SELECT DISTINCT category_name FROM categories
    """)

    run_and_log(cursor, log_lines, "Query 4: IN + BETWEEN (highly-rated, mid-priced books)", """
        SELECT title, rating, price_gbp FROM books
        WHERE rating IN (4, 5) AND price_gbp BETWEEN 20 AND 40
        ORDER BY price_gbp
    """)

    join_sql = """
        SELECT c.category_name, b.title, b.rating
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        WHERE (
            SELECT COUNT(*) FROM books b2
            WHERE b2.category_id = b.category_id AND b2.rating > b.rating
        ) < 3
        ORDER BY c.category_name, b.rating DESC
    """
    run_and_log(cursor, log_lines, "Query 5: JOIN (top 3 rated books per category)", join_sql)

    with open("query_results.txt", "w") as f:
        f.writelines(log_lines)
    print("\nSaved query_results.txt")

    # --- pandas verification (Task 6) ---
    cheap_books_df = pd.read_sql("SELECT title, price_gbp FROM books WHERE price_gbp < 20", conn)
    join_via_sql_df = pd.read_sql(join_sql, conn)

    books_df = pd.read_sql("SELECT * FROM books", conn)
    categories_df = pd.read_sql("SELECT * FROM categories", conn)
    merged_df = pd.merge(books_df, categories_df, on="category_id", how="inner")
    merged_df["rank_in_category"] = (
        merged_df.groupby("category_name")["rating"].rank(method="min", ascending=False)
    )
    join_via_pandas_df = (
        merged_df[merged_df["rank_in_category"] <= 3]
        [["category_name", "title", "rating"]]
        .sort_values(["category_name", "rating"], ascending=[True, False])
        .reset_index(drop=True)
    )

    sql_comparable = join_via_sql_df.sort_values(["category_name", "rating", "title"]).reset_index(drop=True)
    pandas_comparable = join_via_pandas_df.sort_values(["category_name", "rating", "title"]).reset_index(drop=True)
    print(f"\npd.read_sql vs pd.merge match: {sql_comparable.equals(pandas_comparable)}")

    conn.close()


if __name__ == "__main__":
    main()
