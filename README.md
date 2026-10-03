# Zepto Capstone Project

A three-module AI/ML engineering project built around a fictional company, Zepto.

| Module | Folder | What it does |
|---|---|---|
| 1. Data Pipeline | `data_pipeline/` | Scrapes books.toscrape.com, cleans the data, converts GBP to INR (1 GBP = 105.50 INR, fixed), stores it in a two-table SQLite database, and queries it with SQL and pandas |
| 2. Analytics Pipeline | `analytics/` | Titanic EDA, three classifiers, imbalance handling, hyperparameter tuning, a regression side-task, and a saved end-to-end pipeline |
| 3. Support Assistant | `support_assistant/` | A RAG service with ChromaDB, a LangGraph router and a FastAPI `/ask` endpoint, running in deterministic mock mode by default, plus a Dockerfile |

Each folder has its own `README.md` with install and run steps, design decisions and results.

## Repository structure

```
zepto-capstone/
├── data_pipeline/
├── analytics/
└── support_assistant/
```

## Git workflow

Each module was developed on its own feature branch, committed to at least twice, and merged into `main`.

## Where to start

Read `data_pipeline/README.md` first, then `analytics/README.md`, then `support_assistant/README.md`.