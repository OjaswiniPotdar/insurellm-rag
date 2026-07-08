

import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import OpenAI
from pydantic import BaseModel

from app.config.settings import get_settings
from app.database.vector_store import VectorStore
from app.services.synthesizer import Synthesizer

load_dotenv(override=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://insurellm-rag.vercel.app"
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

vec = VectorStore()
settings = get_settings()
openai_client = OpenAI(api_key=settings.openai.api_key)


class QueryRequest(BaseModel):
    question: str



GREETINGS = {"hi", "hello", "hey", "good morning", "good evening", "good afternoon"}


def is_greeting(text: str) -> bool:
    return text.lower().strip().rstrip("!.,") in GREETINGS



@app.post("/query")
def query(request: QueryRequest):

    if is_greeting(request.question):
        return {
            "answer": "Hello! I'm the Insurellm Assistant. Ask me anything about our insurance products, contracts, or company.",
            "enough_context": True,
            "thought_process": []
        }

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



@app.post("/query/stream")
async def query_stream(request: QueryRequest):

    if is_greeting(request.question):
        async def greeting_stream():
            msg = "Hello! I'm the Insurellm Assistant. Ask me anything about our insurance products, contracts, or company."
            for char in msg:
                yield f"data: {json.dumps({'token': char})}\n\n"
            yield f"data: {json.dumps({'done': True, 'enough_context': True, 'thought_process': []})}\n\n"
        return StreamingResponse(greeting_stream(), media_type="text/event-stream")

    results = vec.search(request.question, limit=5)

    if results.empty:
        async def empty_stream():
            msg = "I couldn't find relevant information in the knowledge base."
            for char in msg:
                yield f"data: {json.dumps({'token': char})}\n\n"
            yield f"data: {json.dumps({'done': True, 'enough_context': False, 'thought_process': []})}\n\n"
        return StreamingResponse(empty_stream(), media_type="text/event-stream")

    # Build context
    context_str = "\n\n".join(results["contents"].tolist())

    system_prompt = """
# Role
You are an AI assistant for Insurellm, an insurance company.
Answer questions using information retrieved from the company's internal knowledge base.
Use ONLY the information provided. Do NOT fabricate information.
If the context does not contain enough information, clearly state that.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {request.question}\n\nContext:\n{context_str}"}
    ]

    async def token_stream():
        full_answer = ""
        try:
            stream = openai_client.chat.completions.create(
                model=settings.openai.default_model,
                messages=messages,
                max_tokens=1000,
                stream=True,
            )

            for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                if token:
                    full_answer += token
                    yield f"data: {json.dumps({'token': token})}\n\n"

            # Send done signal with metadata
            yield f"data: {json.dumps({'done': True, 'enough_context': True, 'thought_process': []})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(token_stream(), media_type="text/event-stream")



@app.get("/health")
def health():
    return {"status": "ok"}