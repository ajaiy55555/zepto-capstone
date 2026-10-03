from contextlib import asynccontextmanager

from fastapi import FastAPI

from graph import graph
from ingest import ingest_documents
from schemas import AskRequest, AskResponse


@asynccontextmanager
async def lifespan(app):
    count = ingest_documents()           # embed and store the 8 documents when the server starts
    print(f"Ingested {count} documents into ChromaDB")
    yield


app = FastAPI(title="Zepto Support Assistant", lifespan=lifespan)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    result = graph.invoke({"query": request.query})
    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"],
    )