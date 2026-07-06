# import os
# import re
# import nltk
# from docling.document_converter import DocumentConverter
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import tiktoken

# nltk.download("punkt", quiet=True)

# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# encoding = tiktoken.get_encoding("cl100k_base")


# class DocumentObject:

#     def __init__(self, text, metadata=None):
#         self.text = text
#         self.metadata = metadata or {}

#     def __repr__(self):
#         return (
#             f"DocumentObject("
#             f"id={self.metadata.get('chunk_id')}, "
#             f"tokens={self.metadata.get('token_count')}, "
#             f"text={self.text[:60]!r}...)"
#         )


# def load_pdf(path):

#     converter = DocumentConverter()
#     result = converter.convert(path)

#     text = result.document.export_to_markdown()

#     return text


# def load_all_files(root_folder):

#     documents = []

#     for root, dirs, files in os.walk(root_folder):

#         for file in files:

#             if file.endswith(".pdf") or file.endswith(".md") or file.endswith(".txt"):

#                 path = os.path.join(root, file)

#                 try:

#                     if file.endswith(".pdf"):
#                         text = load_pdf(path)

#                     else:
#                         with open(path, "r", encoding="utf-8") as f:
#                             text = f.read()

#                     documents.append((path, text))

#                 except Exception as e:
#                     print(f"Skipping {path}: {e}")

#     return documents

# def clean_text(text):

#     text = re.sub(r"\[\d+\]", "", text)

#     text = re.sub(r"http\S+", "", text)

#     text = re.sub(r"\*\*", "", text)
#     text = re.sub(r"#+\s*", "", text)

#     text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

#     text = re.sub(r"\n{2,}", "\n", text)

#     text = re.sub(r"[ \t]+", " ", text)

#     return text.strip()


# def token_count(text):
#     return len(encoding.encode(text))


# def semantic_chunk(text, similarity_threshold=0.65, buffer_size=1):

#     sentences = nltk.sent_tokenize(text)

#     if len(sentences) <= 1:
#         return sentences

#     buffered_sentences = [
#         " ".join(sentences[max(0, i-buffer_size): i+buffer_size+1])
#         for i in range(len(sentences))
#     ]

#     embeddings = embedding_model.encode(
#         buffered_sentences,
#         batch_size=64,
#         show_progress_bar=False
#     )

#     chunks = []
#     current_chunk = [sentences[0]]

#     for i in range(1, len(sentences)):

#         sim = cosine_similarity(
#             [embeddings[i-1]],
#             [embeddings[i]]
#         )[0][0]

#         if sim < similarity_threshold and token_count(" ".join(current_chunk)) > 100:

#             chunks.append(" ".join(current_chunk))
#             current_chunk = [sentences[i]]

#         else:
#             current_chunk.append(sentences[i])

#     chunks.append(" ".join(current_chunk))

#     return chunks


# def recursive_chunk(text, chunk_size=600, overlap=100):

#     tokens = encoding.encode(text)

#     chunks = []

#     start = 0

#     while start < len(tokens):

#         end = start + chunk_size

#         chunk_tokens = tokens[start:end]

#         chunk_text = encoding.decode(chunk_tokens)

#         chunks.append(chunk_text)

#         start = end - overlap

#         if start < 0:
#             start = 0

#     return chunks


# def create_chunks(
#         text,
#         similarity_threshold=0.65,
#         max_tokens=600,
#         overlap_tokens=100,
#         source="document"):

#     cleaned = clean_text(text)

#     semantic_chunks = semantic_chunk(
#         cleaned,
#         similarity_threshold=similarity_threshold
#     )

#     final_chunks = []

#     for chunk in semantic_chunks:

#         if token_count(chunk) > max_tokens:

#             recursive_chunks = recursive_chunk(
#                 chunk,
#                 chunk_size=max_tokens,
#                 overlap=overlap_tokens
#             )

#             final_chunks.extend(recursive_chunks)

#         else:

#             final_chunks.append(chunk)

#     docs = []

#     for i, chunk in enumerate(final_chunks):

#         if not chunk.strip():
#             continue

#         docs.append(
#             DocumentObject(
#                 text=chunk,
#                 metadata={
#                     "chunk_id": i,
#                     "token_count": token_count(chunk),
#                     "source": source
#                 }
#             )
#         )

#     return docs

# if __name__ == "__main__":

#     dataset_folder = "knowledge-base"

#     files = load_all_files(dataset_folder)

#     all_chunks = []

#     for path, text in files:

#         docs = create_chunks(
#             text,
#             similarity_threshold=0.85,
#             max_tokens=600,
#             overlap_tokens=100,
#             source=path
#         )

#         all_chunks.extend(docs)

#     print("\n----------------------------------")
#     print(f"Total files processed: {len(files)}")
#     print(f"Total chunks created: {len(all_chunks)}")
#     print("----------------------------------\n")

#     for doc in all_chunks[:5]:

#         print(
#             f"--- Chunk {doc.metadata['chunk_id']} "
#             f"({doc.metadata['token_count']} tokens) ---"
#         )

#         print(doc.text)
#         print()



# import json
# import logging
# import uuid
# from datetime import datetime

# import pandas as pd
# import psycopg2
# from tqdm import tqdm
# from sentence_transformers import SentenceTransformer


# # -------------------------------------------------------------
# # Logging
# # -------------------------------------------------------------

# logging.basicConfig(level=logging.INFO)


# # -------------------------------------------------------------
# # Vector Store Class (pgvector version)
# # -------------------------------------------------------------

# class VectorStore:

#     def __init__(self):

#         self.conn = psycopg2.connect(
#             host="localhost",
#             port=5432,
#             database="rag_db",
#             user="postgres",
#             password="postgres"
#         )

#         self.cursor = self.conn.cursor()

#         self.model = SentenceTransformer("all-MiniLM-L6-v2")

#     # ---------------------------------------------------------

#     def get_embedding(self, text):

#         return self.model.encode(text).tolist()

#     # ---------------------------------------------------------

#     def create_tables(self):

#         self.cursor.execute(
#             """
#            CREATE EXTENSION IF NOT EXISTS vector;

#             CREATE TABLE documents (
#                 id TEXT PRIMARY KEY,
#                 metadata JSONB,
#                 contents TEXT,
#                 embedding VECTOR(1536)
#             );
#             """
#         )

#         self.conn.commit()

#         logging.info("Vector table verified")
#         # ---------------------------------------------------------

#     def create_index(self):

#         self.cursor.execute(
#             """
#             CREATE INDEX IF NOT EXISTS documents_embedding_idx
#             ON documents
#             USING ivfflat (embedding vector_cosine_ops)
#             WITH (lists = 100);
#             """
#         )

#         self.conn.commit()

#         logging.info("Vector index created")

#     # ---------------------------------------------------------

#     def upsert(self, df):

#         for _, row in tqdm(df.iterrows(), total=len(df)):

#             self.cursor.execute(
#     """
#     INSERT INTO documents
#     (id, metadata, contents, embedding)
#     VALUES (%s,%s,%s,%s)
#     ON CONFLICT (id) DO NOTHING
#     """,
#     (
#         row["id"],
#         json.dumps(row["metadata"], default=str),
#         row["contents"],
#         row["embedding"]
#     )
# )

#         self.conn.commit()

#         logging.info("Embeddings inserted into pgvector")


# # -------------------------------------------------------------
# # Initialize Vector Store
# # -------------------------------------------------------------

# vec = VectorStore()


# # -------------------------------------------------------------
# # Convert Chunk → Vector Record
# # -------------------------------------------------------------

# def prepare_record(chunk: dict) -> dict:

#     content = chunk["text"]

#     embedding = vec.get_embedding(content)

#     record = {
#         "id": str(uuid.uuid4()),
#         "metadata": {
#             "source": chunk["source"],
#             "chunk_id": chunk["chunk_id"],
#             "token_count": chunk["token_count"],
#             "created_at": datetime.now()
#         },
#         "contents": content,
#         "embedding": embedding
#     }

#     return record


# # -------------------------------------------------------------
# # Load Chunk File
# # -------------------------------------------------------------

# def load_chunks(file_path="chunks.json"):

#     logging.info(f"Loading chunks from {file_path}")

#     with open(file_path, "r", encoding="utf-8") as f:
#         chunks = json.load(f)

#     logging.info(f"Total chunks loaded: {len(chunks)}")

#     return chunks


# # -------------------------------------------------------------
# # Main Pipeline
# # -------------------------------------------------------------

# if __name__ == "__main__":

#     # Step 1: Load chunks
#     chunks = load_chunks()

#     # Step 2: Convert chunks → records
#     records = []

#     for chunk in chunks:
#         record = prepare_record(chunk)
#         records.append(record)

#     records_df = pd.DataFrame(records)

#     logging.info(f"Total embeddings generated: {len(records_df)}")

#     # Step 3: Create tables
#     vec.create_tables()

#     # Step 4: Create index
#     vec.create_index()

#     # Step 5: Upload embeddings
#     vec.upsert(records_df)

#     logging.info("Embeddings successfully uploaded to pgvector")


import json
import uuid
import logging
from datetime import datetime

import pandas as pd
from app.database.vector_store import VectorStore


logging.basicConfig(level=logging.INFO)

vec = VectorStore()


# ------------------------------------------------
# Load chunks
# ------------------------------------------------

def load_chunks(file_path="chunks.json"):

    logging.info(f"Loading chunks from {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    logging.info(f"Total chunks loaded: {len(chunks)}")

    return chunks


# ------------------------------------------------
# Convert chunk → vector record
# ------------------------------------------------

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


# ------------------------------------------------
# Main pipeline
# ------------------------------------------------

if __name__ == "__main__":

    chunks = load_chunks()

    records = prepare_records(chunks)

    df = pd.DataFrame(records)

    logging.info(f"Total embeddings generated: {len(df)}")

    vec.create_tables()

    vec.create_index()

    vec.upsert(df)

    logging.info("Embeddings successfully stored")