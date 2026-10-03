"""LangGraph flow: classify_intent -> (retrieve_and_answer | direct_answer)."""

import os
from typing import List, TypedDict

import requests
from langgraph.graph import END, StateGraph

from ingest import retrieve
from prompts import build_prompt
from schemas import AskResponse

POLICY_KEYWORDS = [
    "delivery", "return", "refund", "membership",
    "tracking", "cancel", "gift card", "support hours",
]

MAX_RETRIES = 2                          # extra attempts after the first


def is_mock():
    """Mock mode is the default. Only MOCK_LLM=0 switches to a real LLM."""
    return os.getenv("MOCK_LLM", "1") != "0"


def call_llm(prompt):
    """Optional real-LLM call (Groq free tier). Never reached in mock mode."""
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
        json={
            "model": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def generate_validated(prompt):
    """Ask the LLM, check the answer matches the schema, retry up to 2 more times if not."""
    current_prompt = prompt
    last_error = "unknown error"
    for _ in range(1 + MAX_RETRIES):
        try:
            raw = call_llm(current_prompt).strip()
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return AskResponse.model_validate_json(raw)
        except Exception as error:
            last_error = str(error)[:200]
            current_prompt = (
                prompt
                + f"\n\nYour previous output was invalid ({last_error}). "
                + 'Return ONLY a valid JSON object with keys "answer", "sources", "confidence".'
            )
    return AskResponse(
        answer=f"ERROR: could not produce a valid response ({last_error})",
        sources=[],
        confidence=0.0,
    )


class GraphState(TypedDict, total=False):
    query: str
    intent: str
    retrieved: List[dict]
    answer: str
    sources: List[str]
    confidence: float


def classify_intent(state):
    query = state["query"]
    if is_mock():
        lowered = query.lower()
        intent = "policy_question" if any(k in lowered for k in POLICY_KEYWORDS) else "general_question"
    else:
        reply = call_llm(
            "Classify this query as exactly one word: policy_question (about Zepto delivery, returns, "
            "refunds, membership, tracking, cancellation, gift cards or support hours) or "
            f"general_question (anything else).\nQuery: {query}\nAnswer:"
        ).strip().lower()
        intent = "policy_question" if "policy" in reply else "general_question"
    return {"intent": intent}


def retrieve_and_answer(state):
    chunks = retrieve(state["query"], k=3)           # real retrieval in both modes
    sources = [c["id"] for c in chunks]

    if is_mock():
        snippet = chunks[0]["text"][:200]
        return {
            "retrieved": chunks,
            "answer": f"Based on the retrieved context: {snippet}",
            "sources": sources,
            "confidence": 1.0,
        }

    result = generate_validated(build_prompt(state["query"], chunks))
    return {
        "retrieved": chunks,
        "answer": result.answer,
        "sources": result.sources,
        "confidence": result.confidence,
    }


def direct_answer(state):
    if is_mock():
        return {
            "answer": "I can only answer questions about Zepto policies right now.",
            "sources": [],
            "confidence": 1.0,
        }

    result = generate_validated(
        'Answer briefly. Return ONLY JSON with keys "answer" (string), "sources" (empty list), '
        f'"confidence" (0 to 1).\nQuestion: {state["query"]}'
    )
    return {"answer": result.answer, "sources": [], "confidence": result.confidence}


def route(state):
    """Choose the next node. This does not depend on MOCK_LLM."""
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"


builder = StateGraph(GraphState)
builder.add_node("classify_intent", classify_intent)
builder.add_node("retrieve_and_answer", retrieve_and_answer)
builder.add_node("direct_answer", direct_answer)

builder.set_entry_point("classify_intent")
builder.add_conditional_edges(
    "classify_intent",
    route,
    {"retrieve_and_answer": "retrieve_and_answer", "direct_answer": "direct_answer"},
)
builder.add_edge("retrieve_and_answer", END)
builder.add_edge("direct_answer", END)

graph = builder.compile()