from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from openai import OpenAI

from app.database.vector_store import VectorStore
from app.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()



def tokenize(text: str) -> list[str]:
    """Better tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())



@dataclass
class HybridSearchConfig:

    vector_top_k: int = 50
    bm25_top_k: int = 50
    rerank_top_k: int = 10
    final_top_k: int = 3

    semantic_weight: float = 0.7
    bm25_weight: float = 0.3

    rrf_k: int = 60
    use_rrf: bool = True

    use_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"



@dataclass
class RetrievedDoc:

    id: str
    contents: str
    metadata: dict = field(default_factory=dict)

    distance: float = 1.0
    semantic_score: float = 0.0

    bm25_score: float = 0.0
    bm25_norm: float = 0.0

    hybrid_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0




def _rrf_fusion(semantic_docs, bm25_docs, k=60):

    rrf_scores = {}
    index = {}

    for rank, doc in enumerate(semantic_docs, start=1):
        rrf_scores[doc.id] = rrf_scores.get(doc.id, 0) + 1 / (k + rank)
        index[doc.id] = doc

    for rank, doc in enumerate(bm25_docs, start=1):
        rrf_scores[doc.id] = rrf_scores.get(doc.id, 0) + 1 / (k + rank)

        if doc.id not in index:
            index[doc.id] = doc

    for doc_id, score in rrf_scores.items():
        index[doc_id].hybrid_score = score

    return sorted(index.values(), key=lambda d: d.hybrid_score, reverse=True)


def _linear_fusion(semantic_docs, bm25_docs, alpha=0.7):

    def normalize(values):

        lo, hi = min(values, default=0), max(values, default=1)
        rng = hi - lo or 1e-9
        return [(v - lo) / rng for v in values]

    sem_norm = normalize([d.semantic_score for d in semantic_docs])

    for doc, score in zip(semantic_docs, sem_norm):
        doc.semantic_score = score

    bm25_norm = normalize([d.bm25_score for d in bm25_docs])

    for doc, score in zip(bm25_docs, bm25_norm):
        doc.bm25_norm = score

    merged = {d.id: d for d in semantic_docs}

    for doc in bm25_docs:

        if doc.id in merged:
            merged[doc.id].bm25_norm = doc.bm25_norm
        else:
            merged[doc.id] = doc

    for doc in merged.values():
        doc.hybrid_score = alpha * doc.semantic_score + (1 - alpha) * doc.bm25_norm

    return sorted(merged.values(), key=lambda d: d.hybrid_score, reverse=True)




class HybridRetriever:

    def __init__(self, config: HybridSearchConfig | None = None):

        self.config = config or HybridSearchConfig()
        self.vector_store = VectorStore()

        self.reranker = (
            CrossEncoder(self.config.reranker_model)
            if self.config.use_reranker
            else None
        )



    def _semantic_retrieve(self, query):

        logger.info("Running vector search")

        df = self.vector_store.search(query, limit=self.config.vector_top_k)

        if df.empty:
            return []

        docs = []

        for _, row in df.iterrows():

            dist = float(row.get("distance", 1))

            docs.append(
                RetrievedDoc(
                    id=str(row.get("id", row.name)),
                    contents=row["contents"],
                    metadata=row.get("metadata", {}),
                    distance=dist,
                    semantic_score=max(0, 1 - dist),
                )
            )

        return docs

 

    def _bm25_score(self, query, docs):

        logger.info("Running BM25 scoring")

        tokenized_docs = [tokenize(d.contents) for d in docs]

        bm25 = BM25Okapi(tokenized_docs)

        scores = bm25.get_scores(tokenize(query))

        for doc, score in zip(docs, scores):
            doc.bm25_score = float(score)

        docs = sorted(docs, key=lambda d: d.bm25_score, reverse=True)

        return docs[: self.config.bm25_top_k]

   

    def _fuse(self, semantic_docs, bm25_docs):

        if self.config.use_rrf:
            return _rrf_fusion(semantic_docs, bm25_docs, self.config.rrf_k)

        return _linear_fusion(semantic_docs, bm25_docs, self.config.semantic_weight)


    def _rerank(self, query, docs):

        if not self.reranker or not docs:

            for d in docs:
                d.final_score = d.hybrid_score

            return docs

        logger.info("Running cross-encoder reranking")

        pairs = [(query, d.contents) for d in docs]

        scores = self.reranker.predict(pairs, batch_size=32)

        for doc, score in zip(docs, scores):

            doc.rerank_score = float(score)
            doc.final_score = float(score)

        return sorted(docs, key=lambda d: d.rerank_score, reverse=True)


    def hybrid_search(self, query):

        semantic_docs = self._semantic_retrieve(query)

        if not semantic_docs:
            return []

        bm25_docs = self._bm25_score(query, list(semantic_docs))

        fused = self._fuse(semantic_docs, bm25_docs)

        return fused[: self.config.rerank_top_k]


    def retrieve(self, query):

        start = time.time()

        candidates = self.hybrid_search(query)

        final_docs = self._rerank(query, candidates)

        final_docs = final_docs[: self.config.final_top_k]

        logger.info(f"Retrieval latency: {time.time() - start:.2f}s")

        if not final_docs:
            return pd.DataFrame()

        records = []

        for d in final_docs:

            records.append(
                {
                    "id": d.id,
                    "contents": d.contents,
                    "metadata": d.metadata,
                    "semantic_score": round(d.semantic_score, 4),
                    "bm25_score": round(d.bm25_score, 4),
                    "hybrid_score": round(d.hybrid_score, 4),
                    "rerank_score": round(d.rerank_score, 4),
                    "final_score": round(d.final_score, 4),
                }
            )

        return pd.DataFrame.from_records(records)


client_llm = OpenAI(api_key=settings.openai.api_key)

SYSTEM_PROMPT = (
    "You are an insurance assistant. "
    "Answer only using the provided context. "
    "If the context does not contain the answer, say so clearly."
)


def stream_answer(question, context_df):

    context = "\n\n".join(context_df["contents"].tolist())

    stream = client_llm.chat.completions.create(
        model=settings.openai.default_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"},
        ],
        stream=True,
        max_completion_tokens=800,
    )

    print("\nAnswer:\n")

    for chunk in stream:

        delta = chunk.choices[0].delta.content

        if delta:
            print(delta, end="", flush=True)

    print("\n")



retriever = HybridRetriever()

def run_query(question):

    print("\n" + "=" * 56)
    print("Question:", question)
    print("-" * 56)

    results = retriever.retrieve(question)

    if results.empty:
        print("No relevant documents found")
        return

    stream_answer(question, results)

    print("=" * 56 + "\n")


if __name__ == "__main__":

    run_query("What are features of Bizllm?")