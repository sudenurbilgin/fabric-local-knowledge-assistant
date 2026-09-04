import math
import sys

from foundry_local_sdk import Configuration, FoundryLocalManager

from .config import (
    CPU_EXECUTION_PROVIDER,
    DOCUMENTS_DIRECTORY,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL_ALIAS,
    FOUNDRY_APP_NAME,
)
from .documents import load_documents, process_documents


def prepare_chunks():
    documents = load_documents(DOCUMENTS_DIRECTORY)
    processed_documents = process_documents(documents)
    chunks = [
        chunk
        for document in processed_documents
        for chunk in document["chunks"]
    ]

    if not chunks:
        raise RuntimeError("The document collection produced no chunks to embed.")

    return documents, chunks


def generate_chunk_embeddings(embedding_client, chunks):
    embedded_chunks = []
    batch_count = math.ceil(len(chunks) / EMBEDDING_BATCH_SIZE)

    for batch_number, start in enumerate(
        range(0, len(chunks), EMBEDDING_BATCH_SIZE), start=1
    ):
        batch = chunks[start : start + EMBEDDING_BATCH_SIZE]
        print(f"Embedding batch {batch_number}/{batch_count}...", flush=True)

        response = embedding_client.generate_embeddings(
            [chunk["text"] for chunk in batch]
        )
        response_items = sorted(response.data, key=lambda item: item.index)

        if len(response_items) != len(batch):
            raise RuntimeError(
                f"Batch {batch_number} contained {len(batch)} chunks but returned "
                f"{len(response_items)} embeddings."
            )

        embedded_chunks.extend(
            {
                **chunk,
                "embedding": response_item.embedding,
            }
            for chunk, response_item in zip(batch, response_items)
        )

    return embedded_chunks


def select_embedding_model(manager):
    embedding_model = manager.catalog.get_model(EMBEDDING_MODEL_ALIAS)
    if embedding_model is None:
        raise RuntimeError("Embedding model was not found in the Foundry Local catalog.")

    cpu_variant = next(
        variant
        for variant in embedding_model.variants
        if variant.info.runtime
        and variant.info.runtime.execution_provider == CPU_EXECUTION_PROVIDER
    )
    embedding_model.select_variant(cpu_variant)
    return embedding_model


def validate_embeddings(chunks, embedded_chunks):
    if len(embedded_chunks) != len(chunks):
        raise RuntimeError("The number of embeddings does not match the chunk count.")

    if not all(item["embedding"] for item in embedded_chunks):
        raise RuntimeError("At least one embedding is empty.")

    dimensions = {len(item["embedding"]) for item in embedded_chunks}
    if len(dimensions) != 1:
        raise RuntimeError("Embedding dimensions are not consistent.")

    metadata_preserved = all(
        embedded["source"] == original["source"]
        and embedded["chunk_number"] == original["chunk_number"]
        and embedded["text"] == original["text"]
        for original, embedded in zip(chunks, embedded_chunks)
    )
    if not metadata_preserved:
        raise RuntimeError("Chunk metadata was not preserved.")

    all_values_finite = all(
        math.isfinite(value)
        for item in embedded_chunks
        for value in item["embedding"]
    )
    if not all_values_finite:
        raise RuntimeError("At least one embedding contains NaN or infinity.")

    return dimensions.pop(), metadata_preserved, all_values_finite


def text_preview(text, max_chars=240):
    compact_text = " ".join(text.split())
    if len(compact_text) <= max_chars:
        return compact_text
    return compact_text[:max_chars].rstrip() + "..."


def print_example(number, item):
    print(f"\nExample {number}:")
    print(f"SOURCE: {item['source']}")
    print(f"CHUNK: {item['chunk_number']}")
    print(f"TEXT PREVIEW: {text_preview(item['text'])}")
    print(f"EMBEDDING DIMENSIONS: {len(item['embedding'])}")
    print(f"FIRST 5 EMBEDDING VALUES: {item['embedding'][:5]}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    documents, chunks = prepare_chunks()
    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks prepared: {len(chunks)}")

    config = Configuration(app_name=FOUNDRY_APP_NAME)
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

        print(f"Embeddings generated: {len(embedded_chunks)}")
        print(f"Embedding dimensions: {dimensions}")

        fabric_example = next(
            item
            for item in embedded_chunks
            if item["source"] == "microsoft-fabric-overview.md"
            and item["chunk_number"] == 1
        )
        data_factory_chunks = [
            item
            for item in embedded_chunks
            if item["source"] == "data-factory-overview.md"
        ]
        data_factory_example = data_factory_chunks[len(data_factory_chunks) // 2]
        direct_lake_example = next(
            item
            for item in embedded_chunks
            if item["source"] == "direct-lake-overview.md"
            and item["chunk_number"] == 1
        )

        print_example(1, fabric_example)
        print_example(2, data_factory_example)
        print_example(3, direct_lake_example)

        print(f"\nDimensions consistent: {dimensions > 0}")
        print(f"All numeric values finite: {all_values_finite}")
        print(f"Chunk metadata preserved: {metadata_preserved}")
        print(f"Batch size: {EMBEDDING_BATCH_SIZE}")
    finally:
        embedding_model.unload()


if __name__ == "__main__":
    main()
