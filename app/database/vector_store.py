import logging
import os
import time
from typing import List, Union

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.config.settings import get_settings

load_dotenv(override=True)


class VectorStore:

    def __init__(self):

        self.settings = get_settings()

        self.openai_client = OpenAI(
            api_key=self.settings.openai.api_key
        )

        self.embedding_model = self.settings.openai.embedding_model

        self.vector_settings = self.settings.vector_store

        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=60,
        )

        self._ensure_collection()

        logging.info(f"Qdrant collection '{self.vector_settings.table_name}' ready")




    def _ensure_collection(self):

        existing = [c.name for c in self.client.get_collections().collections]

        if self.vector_settings.table_name not in existing:

            self.client.create_collection(
                collection_name=self.vector_settings.table_name,
                vectors_config=VectorParams(
                    size=self.vector_settings.embedding_dimensions,
                    distance=Distance.COSINE,
                ),
            )

            logging.info(
                f"Created Qdrant collection '{self.vector_settings.table_name}'"
            )


    def create_tables(self):
        self._ensure_collection()


    def create_index(self):
        logging.info("Qdrant manages indexing automatically (HNSW)")


    def drop_index(self):
        logging.info("Qdrant manages indexes automatically")



    def get_embedding(self, text: str) -> List[float]:

        text = text.replace("\n", " ")

        start_time = time.time()

        embedding = (
            self.openai_client.embeddings.create(
                input=[text],
                model=self.embedding_model
            )
            .data[0]
            .embedding
        )

        elapsed = time.time() - start_time

        logging.info(f"Embedding generated in {elapsed:.3f} seconds")

        return list(embedding)



    def upsert(self, df: pd.DataFrame):

        """
        Expected DataFrame schema:
        id, metadata, contents, embedding
        """

        points = []

        for i, (_, row) in enumerate(df.iterrows()):

            meta = row["metadata"]

            if not isinstance(meta, dict):
                meta = {"info": str(meta)}

            payload = {"contents": str(row["contents"])}
            payload.update({k: str(v) for k, v in meta.items()})

            points.append(
                PointStruct(
                    id=i,
                    vector=list(row["embedding"]),
                    payload=payload,
                )
            )

     
        batch_size = 10
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.vector_settings.table_name,
                points=batch,
            )
            logging.info(
                f"Uploaded batch {i//batch_size + 1} "
                f"({len(batch)} points, total {min(i+batch_size, len(points))}/{len(points)})"
            )

        logging.info(
            f"Inserted {len(df)} records into '{self.vector_settings.table_name}'"
        )


    def search(
        self,
        query_text: str,
        limit: int = 5,
        metadata_filter: Union[dict, List[dict]] = None,
        predicates=None,
        time_range=None,
        return_dataframe: bool = True,
    ):

        query_embedding = self.get_embedding(query_text)

        start_time = time.time()

        query_filter = None

        if metadata_filter and isinstance(metadata_filter, dict):
            conditions = [
                FieldCondition(
                    key=k,
                    match=MatchValue(value=str(v))
                )
                for k, v in metadata_filter.items()
            ]
            query_filter = Filter(must=conditions)

        results = self.client.query_points(
            collection_name=self.vector_settings.table_name,
            query=query_embedding,
            limit=limit,
            with_payload=True,
            query_filter=query_filter,
        ).points

        elapsed = time.time() - start_time

        logging.info(f"Vector search completed in {elapsed:.3f} seconds")

        if return_dataframe:
            return self._create_dataframe_from_results(results)

        return results




    def _create_dataframe_from_results(self, results) -> pd.DataFrame:

        rows = []

        for hit in results:

            row = {
                "id": str(hit.id),
                "distance": 1 - hit.score,
            }

            if hit.payload:
                row.update(hit.payload)

            rows.append(row)

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows)


    

    def delete(
        self,
        ids: List[str] = None,
        metadata_filter: dict = None,
        delete_all: bool = False,
    ):

        if sum(bool(x) for x in (ids, metadata_filter, delete_all)) != 1:
            raise ValueError(
                "Provide exactly one of: ids, metadata_filter, delete_all"
            )

        if delete_all:

            self.client.delete_collection(self.vector_settings.table_name)
            self._ensure_collection()

            logging.info(
                f"Deleted all records from '{self.vector_settings.table_name}'"
            )

        elif ids:

            from qdrant_client.models import PointIdsList

            self.client.delete(
                collection_name=self.vector_settings.table_name,
                points_selector=PointIdsList(points=[int(i) for i in ids]),
            )

            logging.info(f"Deleted {len(ids)} records")

        elif metadata_filter:

            conditions = [
                FieldCondition(key=k, match=MatchValue(value=str(v)))
                for k, v in metadata_filter.items()
            ]

            self.client.delete(
                collection_name=self.vector_settings.table_name,
                points_selector=Filter(must=conditions),
            )

            logging.info("Deleted records matching metadata filter")