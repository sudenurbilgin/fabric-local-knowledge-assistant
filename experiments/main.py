import math

from foundry_local_sdk import Configuration, FoundryLocalManager


documents = [
    "Foundry Local runs AI models directly on your device without cloud connectivity.",
    "The Foundry Local SDK supports Python, C#, JavaScript, and Rust.",
    "Embedding models convert text into numerical vectors for similarity search.",
    "Foundry Local uses ONNX Runtime for efficient model inference on CPUs and GPUs.",
    "The model catalog provides pre-optimized models that you can download and run locally.",
    "Retrieval-augmented generation grounds model responses in your own data.",
    "Vector similarity search finds documents that are semantically close to a query.",
    "Chat completions generate natural language responses from a prompt and context.",
]


def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(y * y for y in b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def find_relevant(query_embedding, doc_embeddings, top_k=2):
    results = [
        (index, cosine_similarity(query_embedding, doc_embedding))
        for index, doc_embedding in enumerate(doc_embeddings)
    ]
    results.sort(key=lambda result: result[1], reverse=True)
    return results[:top_k]


def main():
    config = Configuration(app_name="foundry_local_rag")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    if embedding_model is None:
        raise RuntimeError("Embedding model was not found in the Foundry Local catalog.")

    cpu_variant = next(
        variant
        for variant in embedding_model.variants
        if variant.info.runtime
        and variant.info.runtime.execution_provider == "CPUExecutionProvider"
    )
    embedding_model.select_variant(cpu_variant)

    embedding_model.download(
        lambda progress: print(
            f"\rDownloading embedding model: {progress:.1f}%", end="", flush=True
        )
    )
    print()
    embedding_model.load()

    try:
        embedding_client = embedding_model.get_embedding_client()

        response = embedding_client.generate_embeddings(documents)
        doc_embeddings = [item.embedding for item in response.data]
        print(f"Indexed {len(doc_embeddings)} documents.")

        query = "How does Foundry Local work without the cloud?"
        query_response = embedding_client.generate_embedding(query)
        query_embedding = query_response.data[0].embedding

        matches = find_relevant(query_embedding, doc_embeddings, top_k=2)

        print(f"\nQuery: {query}\n")
        print("Top matches:\n")

        for rank, (document_index, score) in enumerate(matches, start=1):
            print(f"{rank}. Score: {score:.4f}")
            print(f"   {documents[document_index]}\n")
    finally:
        embedding_model.unload()


if __name__ == "__main__":
    main()
