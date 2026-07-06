import json
import uuid
import logging
from datetime import datetime

import pandas as pd
from app.database.vector_store import VectorStore


logging.basicConfig(level=logging.INFO)

vec = VectorStore()


def load_chunks(file_path="chunks.json"):

    logging.info(f"Loading chunks from {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    logging.info(f"Total chunks loaded: {len(chunks)}")

    return chunks


def prepare_records(chunks):

    records = []

    for chunk in chunks:

        embedding = vec.get_embedding(chunk["text"])

        records.append(
            {
                "id": str(uuid.uuid4()),
                "metadata": {
                    "source": chunk["source"],
                    "chunk_id": chunk["chunk_id"],
                    "token_count": chunk["token_count"],
                    "created_at": datetime.now().isoformat()
                },
                "contents": chunk["text"],
                "embedding": embedding
            }
        )

    return records



if __name__ == "__main__":

    chunks = load_chunks()

    records = prepare_records(chunks)

    df = pd.DataFrame(records)

    logging.info(f"Total embeddings generated: {len(df)}")

    vec.create_tables()

    vec.create_index()

    vec.upsert(df)

    logging.info("Embeddings successfully stored in Qdrant")