import json
import sqlite3
import sys

from foundry_local_sdk import Configuration, FoundryLocalManager

from .config import (
    DATABASE_PATH,
    EMBEDDING_BATCH_SIZE,
    EXPECTED_EMBEDDING_DIMENSIONS,
    FOUNDRY_APP_NAME,
)
from .embeddings import (
    generate_chunk_embeddings,
    prepare_chunks,
    select_embedding_model,
    text_preview,
    validate_embeddings,
)


def generate_embeddings_for_chunks(chunks):
    config = Configuration(app_name=FOUNDRY_APP_NAME)
    if FoundryLocalManager.instance is None:
        FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    embedding_model = select_embedding_model(manager)

    embedding_model.download(
        lambda progress: print(
            f"\rDownloading embedding model: {progress:.1f}%", end="", flush=True
        )
    )
    print()
    embedding_model.load()

    try:
        embedding_client = embedding_model.get_embedding_client()
        embedded_chunks = generate_chunk_embeddings(embedding_client, chunks)
        dimensions, metadata_preserved, all_values_finite = validate_embeddings(
            chunks, embedded_chunks
        )
    finally:
        embedding_model.unload()

    if dimensions != EXPECTED_EMBEDDING_DIMENSIONS:
        raise RuntimeError(
            f"Expected {EXPECTED_EMBEDDING_DIMENSIONS} embedding dimensions, "
            f"but found {dimensions}."
        )

    return (
        embedded_chunks,
        dimensions,
        metadata_preserved,
        all_values_finite,
    )


def report_progress(progress_callback, stage):
    if progress_callback is not None:
        progress_callback(stage)


def generate_embedded_chunks(progress_callback=None):
    report_progress(progress_callback, "preparing_documents")
    report_progress(progress_callback, "extracting_text")
    documents, chunks = prepare_chunks()
    report_progress(progress_callback, "generating_embeddings")
    (
        embedded_chunks,
        dimensions,
        metadata_preserved,
        all_values_finite,
    ) = generate_embeddings_for_chunks(chunks)

    return (
        documents,
        embedded_chunks,
        dimensions,
        metadata_preserved,
        all_values_finite,
    )


def create_chunks_table(connection):
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            chunk_number INTEGER NOT NULL,
            text TEXT NOT NULL,
            embedding TEXT NOT NULL,
            UNIQUE(source, chunk_number)
        )
        """
    )
    connection.commit()


def rebuild_database(
    embedded_chunks,
    database_path=DATABASE_PATH,
    validation_callback=None,
):
    if not embedded_chunks:
        raise RuntimeError("Cannot rebuild the knowledge base without chunks.")

    expected_chunk_count = len(embedded_chunks)
    expected_source_count = len(
        {item["source"] for item in embedded_chunks}
    )
    if expected_source_count == 0:
        raise RuntimeError("Cannot rebuild the knowledge base without source filenames.")

    connection = sqlite3.connect(database_path)
    try:
        create_chunks_table(connection)

        try:
            connection.execute("BEGIN")
            connection.execute("DELETE FROM chunks")
            connection.executemany(
                """
                INSERT INTO chunks (source, chunk_number, text, embedding)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        item["source"],
                        item["chunk_number"],
                        item["text"],
                        json.dumps(item["embedding"]),
                    )
                    for item in embedded_chunks
                ],
            )
            if validation_callback is not None:
                validation_callback()
            validation, examples = validate_database(
                connection,
                expected_chunk_count=expected_chunk_count,
                expected_source_count=expected_source_count,
            )
        except Exception:
            connection.rollback()
            raise
        else:
            connection.commit()
    finally:
        connection.close()

    return validation, examples


def validate_database(
    connection,
    expected_chunk_count,
    expected_source_count,
):
    row_count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    distinct_sources = connection.execute(
        "SELECT COUNT(DISTINCT source) FROM chunks"
    ).fetchone()[0]
    empty_value_count = connection.execute(
        """
        SELECT COUNT(*)
        FROM chunks
        WHERE source IS NULL OR TRIM(source) = ''
           OR text IS NULL OR TRIM(text) = ''
           OR embedding IS NULL OR TRIM(embedding) = ''
        """
    ).fetchone()[0]
    duplicate_count = connection.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT source, chunk_number
            FROM chunks
            GROUP BY source, chunk_number
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    stored_rows = connection.execute(
        "SELECT id, source, chunk_number, text, embedding FROM chunks ORDER BY id"
    ).fetchall()
    embeddings_deserialize = True
    embedding_dimensions_valid = True

    for row in stored_rows:
        try:
            embedding = json.loads(row[4])
        except (json.JSONDecodeError, TypeError):
            embeddings_deserialize = False
            embedding_dimensions_valid = False
            continue

        if not isinstance(embedding, list) or len(embedding) != EXPECTED_EMBEDDING_DIMENSIONS:
            embedding_dimensions_valid = False

    if row_count != expected_chunk_count:
        raise RuntimeError(
            f"Expected {expected_chunk_count} stored rows for this build, "
            f"but found {row_count}."
        )
    if distinct_sources != expected_source_count:
        raise RuntimeError(
            f"Expected {expected_source_count} distinct source filenames for this build, "
            f"but found {distinct_sources}."
        )
    if empty_value_count != 0:
        raise RuntimeError("At least one stored required value is NULL or empty.")
    if duplicate_count != 0:
        raise RuntimeError("Duplicate source and chunk-number pairs were found.")
    if not embeddings_deserialize:
        raise RuntimeError("At least one stored embedding is not valid JSON.")
    if not embedding_dimensions_valid:
        raise RuntimeError(
            "At least one stored embedding does not have "
            f"{EXPECTED_EMBEDDING_DIMENSIONS} values."
        )

    example_rows = connection.execute(
        """
        SELECT id, source, chunk_number, text, embedding
        FROM chunks
        ORDER BY id
        LIMIT 3
        """
    ).fetchall()
    examples = [
        {
            "id": row[0],
            "source": row[1],
            "chunk_number": row[2],
            "text": row[3],
            "embedding": json.loads(row[4]),
        }
        for row in example_rows
    ]

    validation = {
        "row_count": row_count,
        "distinct_sources": distinct_sources,
        "empty_value_count": empty_value_count,
        "duplicate_count": duplicate_count,
        "embeddings_deserialize": embeddings_deserialize,
        "embedding_dimensions_valid": embedding_dimensions_valid,
    }
    return validation, examples


def get_knowledge_base_statistics(database_path=DATABASE_PATH):
    database_uri = database_path.resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(database_uri, uri=True)
    try:
        source_count, chunk_count = connection.execute(
            "SELECT COUNT(DISTINCT source), COUNT(*) FROM chunks"
        ).fetchone()
    finally:
        connection.close()

    return {
        "document_count": source_count,
        "chunk_count": chunk_count,
    }


def persistent_row_count(database_path=DATABASE_PATH):
    return get_knowledge_base_statistics(database_path)["chunk_count"]


def print_example(example):
    print(f"ID: {example['id']}")
    print(f"SOURCE: {example['source']}")
    print(f"CHUNK: {example['chunk_number']}")
    print(f"TEXT PREVIEW: {text_preview(example['text'])}")
    print(f"EMBEDDING DIMENSIONS: {len(example['embedding'])}")
    print(f"FIRST 5 EMBEDDING VALUES: {example['embedding'][:5]}")
    print()


def build_knowledge_base(
    database_path=DATABASE_PATH,
    progress_callback=None,
):
    (
        documents,
        embedded_chunks,
        dimensions,
        metadata_preserved,
        all_values_finite,
    ) = generate_embedded_chunks(progress_callback=progress_callback)
    report_progress(progress_callback, "rebuilding_local_index")
    validation, examples = rebuild_database(
        embedded_chunks,
        database_path=database_path,
        validation_callback=lambda: report_progress(
            progress_callback,
            "validating",
        ),
    )
    statistics = get_knowledge_base_statistics(database_path)
    persisted_count = statistics["chunk_count"]

    if persisted_count != len(embedded_chunks):
        raise RuntimeError(
            f"Expected {len(embedded_chunks)} rows after reopening the database, "
            f"but found {persisted_count}."
        )
    if statistics["document_count"] != len(documents):
        raise RuntimeError(
            f"Expected {len(documents)} sources after reopening the database, "
            f"but found {statistics['document_count']}."
        )

    report_progress(progress_callback, "complete")
    return {
        "documents": documents,
        "embedded_chunks": embedded_chunks,
        "dimensions": dimensions,
        "metadata_preserved": metadata_preserved,
        "all_values_finite": all_values_finite,
        "validation": validation,
        "examples": examples,
        "statistics": statistics,
        "database_path": database_path,
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    result = build_knowledge_base()
    documents = result["documents"]
    embedded_chunks = result["embedded_chunks"]
    dimensions = result["dimensions"]
    metadata_preserved = result["metadata_preserved"]
    all_values_finite = result["all_values_finite"]
    validation = result["validation"]
    examples = result["examples"]
    persisted_count = result["statistics"]["chunk_count"]
    database_path = result["database_path"]

    print("\nKnowledge base built successfully.\n")
    print(f"Documents processed: {len(documents)}")
    print(f"Chunks generated: {len(embedded_chunks)}")
    print(f"Embedding dimensions: {dimensions}")
    print(f"Database: {database_path.name}\n")

    for example in examples:
        print_example(example)

    print(f"Persistent row count after reopening database: {persisted_count}")
    print(f"Distinct source count: {validation['distinct_sources']}")
    print(f"Duplicate count: {validation['duplicate_count']}")
    print(
        "All embeddings deserialize correctly: "
        f"{validation['embeddings_deserialize']}"
    )
    print(
        f"All stored embeddings have {EXPECTED_EMBEDDING_DIMENSIONS} dimensions: "
        f"{validation['embedding_dimensions_valid']}"
    )
    print(f"All embedding values finite before storage: {all_values_finite}")
    print(f"Chunk metadata preserved before storage: {metadata_preserved}")
    print(f"Batch size: {EMBEDDING_BATCH_SIZE}")
    print(f"Database size: {database_path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
