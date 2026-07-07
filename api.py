from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.database.vector_store import VectorStore
from app.services.synthesizer import Synthesizer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://insurellm-rag.vercel.app",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

vec = VectorStore()


class QueryRequest(BaseModel):
    question: str


@app.post("/query")
def query(request: QueryRequest):
    results = vec.search(request.question, limit=5)

    if results.empty:
        return {
            "answer": "I couldn't find relevant information in the knowledge base.",
            "enough_context": False,
            "thought_process": []
        }

    response = Synthesizer.generate_response(
        question=request.question,
        context=results
    )

    return {
        "answer": response.answer,
        "enough_context": response.enough_context,
        "thought_process": response.thought_process
    }


@app.get("/health")
def health():
    return {"status": "ok"}