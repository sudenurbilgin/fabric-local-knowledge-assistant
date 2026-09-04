import json
import math
import sqlite3
import sys

from foundry_local_sdk import Configuration, FoundryLocalManager

from .config import (
    DATABASE_PATH,
    EXPECTED_EMBEDDING_DIMENSIONS,
    FOUNDRY_APP_NAME,
    RETRIEVAL_TOP_K,
)
from .embeddings import select_embedding_model, text_preview


def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(y * y for y in b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def load_stored_chunks():
    database_uri = DATABASE_PATH.resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(database_uri, uri=True)
    try:
        rows = connection.execute(
            """
            SELECT id, source, chunk_number, text, embedding
            FROM chunks
            ORDER BY id
            """
        ).fetchall()
    finally:
        connection.close()

    if not rows:
        raise RuntimeError("The knowledge base contains no indexed chunks.")

    stored_chunks = []
    for row in rows:
        try:
            embedding = json.loads(row[4])
        except (json.JSONDecodeError, TypeError) as error:
            raise RuntimeError(f"Stored embedding for chunk ID {row[0]} is invalid JSON.") from error

        if not isinstance(embedding, list) or len(embedding) != EXPECTED_EMBEDDING_DIMENSIONS:
            raise RuntimeError(
                f"Stored embedding for chunk ID {row[0]} does not have "
                f"{EXPECTED_EMBEDDING_DIMENSIONS} dimensions."
            )
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in embedding):
            raise RuntimeError(
                f"Stored embedding for chunk ID {row[0]} contains an invalid value."
            )

        stored_chunks.append(
            {
                "id": row[0],
                "source": row[1],
                "chunk_number": row[2],
                "text": row[3],
                "embedding": embedding,
            }
        )

    return stored_chunks


def resolve_embedding_model():
    config = Configuration(app_name=FOUNDRY_APP_NAME)
    if FoundryLocalManager.instance is None:
        FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance
    return select_embedding_model(manager)


def download_embedding_model(embedding_model):
    embedding_model.download(
        lambda progress: print(
            f"\rDownloading embedding model: {progress:.1f}%", end="", flush=True
        )
    )
    print()


def validate_query_embedding(query_embedding):
    if len(query_embedding) != EXPECTED_EMBEDDING_DIMENSIONS:
        raise RuntimeError(
            f"Expected a {EXPECTED_EMBEDDING_DIMENSIONS}-dimensional query embedding, "
            f"but received {len(query_embedding)} dimensions."
        )
    if not all(math.isfinite(value) for value in query_embedding):
        raise RuntimeError("The query embedding contains NaN or infinity.")

    return query_embedding


def generate_query_embedding(query):
    embedding_model = resolve_embedding_model()

    download_embedding_model(embedding_model)
    embedding_model.load()

    try:
        embedding_client = embedding_model.get_embedding_client()
        response = embedding_client.generate_embedding(query)
        query_embedding = response.data[0].embedding
    finally:
        embedding_model.unload()

    return validate_query_embedding(query_embedding)


def rank_chunks(query_embedding, stored_chunks, top_k):
    results = [
        {
            "id": chunk["id"],
            "source": chunk["source"],
            "chunk_number": chunk["chunk_number"],
            "text": chunk["text"],
            "similarity": cosine_similarity(query_embedding, chunk["embedding"]),
        }
        for chunk in stored_chunks
    ]
    results.sort(key=lambda result: result["similarity"], reverse=True)
    return results[:top_k]


class PersistentRetriever:
    def __init__(self):
        self._stored_chunks = None
        self._embedding_model = None
        self._embedding_client = None
        self._closed = True

        stored_chunks = load_stored_chunks()
        embedding_model = resolve_embedding_model()
        model_loaded = False

        try:
            download_embedding_model(embedding_model)
            embedding_model.load()
            model_loaded = True
            embedding_client = embedding_model.get_embedding_client()
        except Exception:
            if model_loaded:
                embedding_model.unload()
            else:
                try:
                    embedding_model.unload()
                except Exception:
                    pass
            raise

        self._stored_chunks = stored_chunks
        self._embedding_model = embedding_model
        self._embedding_client = embedding_client
        self._closed = False

    def retrieve(self, query, top_k=RETRIEVAL_TOP_K):
        if self._closed:
            raise RuntimeError("The persistent retriever is closed.")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        response = self._embedding_client.generate_embedding(query)
        query_embedding = validate_query_embedding(response.data[0].embedding)
        return rank_chunks(query_embedding, self._stored_chunks, top_k)

    def close(self):
        if self._closed:
            return

        embedding_model = self._embedding_model
        self._embedding_client = None
        self._embedding_model = None
        self._stored_chunks = None
        self._closed = True

        if embedding_model is not None and embedding_model.is_loaded:
            embedding_model.unload()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.close()


def get_top_chunks(query, top_k=RETRIEVAL_TOP_K):
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    stored_chunks = load_stored_chunks()
    query_embedding = generate_query_embedding(query)
    return rank_chunks(query_embedding, stored_chunks, top_k)


def main():
    from .knowledge_base import get_knowledge_base_statistics

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    query = "What is Direct Lake in Microsoft Fabric?"
    top_k = RETRIEVAL_TOP_K
    results = get_top_chunks(query, top_k=top_k)

    print("QUERY:")
    print(query)
    print(f"\nTOP {top_k} RETRIEVED CHUNKS\n")

    for rank, result in enumerate(results, start=1):
        print(f"RANK: {rank}")
        print(f"SIMILARITY: {result['similarity']:.6f}")
        print(f"SOURCE: {result['source']}")
        print(f"CHUNK: {result['chunk_number']}")
        print(f"TEXT PREVIEW: {text_preview(result['text'], max_chars=400)}")
        print()

    statistics = get_knowledge_base_statistics()
    print(f"Retrieved rows checked: {statistics['chunk_count']}")
    print(f"Query embedding dimensions: {EXPECTED_EMBEDDING_DIMENSIONS}")


if __name__ == "__main__":
    main()
