"""Structured prompt: role - context - task - format - length, plus a negative rule and a few-shot example."""

ROLE = "You are Zepto's customer support assistant. You answer policy questions accurately and politely."

TASK = (
    "Answer the customer's question using only the context provided below. "
    "Do not answer using information that is not present in the provided context. "
    "If the context does not contain the answer, say you do not have that information."
)

FORMAT = (
    "Respond with a single JSON object and nothing else, with exactly these keys: "
    '"answer" (string), "sources" (list of document ids used, e.g. ["doc_01"]), '
    '"confidence" (a number between 0 and 1).'
)

LENGTH = "Keep the answer to at most 3 sentences."

FEW_SHOT = """Example
Context:
[doc_08] Phone support is not offered.
Question: Can I call Zepto support?
Output: {"answer": "No, phone support is not offered. You can use in-app chat or email.", "sources": ["doc_08"], "confidence": 0.9}"""


def build_prompt(question, chunks):
    """Assemble the full prompt, labelling each chunk with its id."""
    context = "\n".join(f"[{c['id']}] {c['text']}" for c in chunks)
    return (
        f"ROLE:\n{ROLE}\n\n"
        f"TASK:\n{TASK}\n\n"
        f"FORMAT:\n{FORMAT}\n\n"
        f"LENGTH:\n{LENGTH}\n\n"
        f"{FEW_SHOT}\n\n"
        f"Now answer for real.\n"
        f"Context:\n{context}\n"
        f"Question: {question}\n"
        f"Output:"
    )