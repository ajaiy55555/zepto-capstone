# Module 3: Zepto Support Assistant

A small RAG (retrieval-augmented generation) service that answers questions about Zepto policies.

## Install and run

```bash
pip install -r requirements.txt
uvicorn main:app --port 8000
```

Open http://127.0.0.1:8000/docs and use POST /ask. `MOCK_LLM` is left at its default (mock mode), so no API key or network call to any LLM is needed.

## Example calls (MOCK_LLM at default)

Request that triggers retrieval:
```json
{"query": "What is your refund policy?"}
```
Response:
```json
{
  "answer": "Based on the retrieved context: Grocery and perishable items may be reported for a return within 24 hours of delivery if damaged, spoiled, or incorrect; non-perishable packaged items may be returned within 7 days of delivery in unop",
  "sources": [
    "doc_02",
    "doc_06",
    "doc_05"
  ],
  "confidence": 1.0
}
```

Request that does not trigger retrieval:
```json
{"query": "Who won the cricket match?"}
```
Response:
```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

## Docker

```bash
docker build -t zepto-support .
docker run -p 7860:7860 zepto-support
```
The service is then available at http://127.0.0.1:7860/docs (POST /ask).

## Architecture: ingestion -> embedding -> retrieval -> generation

1. **Ingestion:** `ingest.py` -> `ingest_documents()` reads `docs/doc_01.txt ... doc_08.txt`. Each document is short, so one document is one chunk.
2. **Embedding:** the same function embeds each chunk locally with `all-MiniLM-L6-v2` (sentence-transformers) and stores the vectors in the ChromaDB collection `zepto_policies` (cosine similarity).
3. **Retrieval:** the `retrieve_and_answer` node in `graph.py` calls `retrieve()` in `ingest.py`, which embeds the query and returns the top 3 most similar chunks. This always runs for real.
4. **Generation:** the `retrieve_and_answer` node produces the final answer for policy questions, using the prompt template in `prompts.py` in real-LLM mode. The `direct_answer` node handles general questions.

Data flow: `POST /ask` (main.py) -> `classify_intent` -> conditional edge -> `retrieve_and_answer` or `direct_answer` -> Pydantic `AskResponse` (answer, sources, confidence) -> JSON.

## MOCK_LLM toggle

The generation step in all three nodes branches on `MOCK_LLM`.
- **Default (unset or `1`, graded):** `classify_intent` uses a keyword check; `retrieve_and_answer` returns "Based on the retrieved context: <first 200 characters of the top chunk>"; `direct_answer` returns a fixed string. Confidence is 1.0. No network call is made.
- **`MOCK_LLM=0` (optional):** the nodes call a real LLM (Groq free tier, key in the `GROQ_API_KEY` environment variable). The structured prompt in `prompts.py` is used, and invalid output is retried up to 2 more times before a clearly marked error response is returned.

## Notes

The mock classifier is a plain keyword match, so a query without any listed keyword (for example "Is phone support available?") is treated as a general question.