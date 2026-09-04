import re
import sys
import time

from foundry_local_sdk import Configuration, FoundryLocalManager

from fabric_rag.rag import build_context, build_messages, select_chat_variant
from fabric_rag.retrieval import get_top_chunks


QUESTIONS = [
    "What is OneLake?",
    "What is Data Factory in Microsoft Fabric?",
    "How does Direct Lake work?",
    "Who is the CEO of Microsoft?",
]
MODEL_ALIASES = [
    "phi-4-mini",
    "qwen3.5-4b",
    "ministral-3-3b-instruct-2512",
]
CITATION_PATTERN = re.compile(r"\[([1-3])\]")


def retrieve_shared_contexts():
    shared_inputs = {}
    for question in QUESTIONS:
        retrieved_chunks = get_top_chunks(question, top_k=3)
        shared_inputs[question] = {
            "retrieved_chunks": retrieved_chunks,
            "messages": build_messages(
                question,
                build_context(retrieved_chunks),
            ),
        }
    return shared_inputs


def test_model(manager, alias, shared_inputs):
    model = manager.catalog.get_model(alias)
    if model is None:
        raise RuntimeError(f"Model alias '{alias}' was not found in the catalog.")

    variant = select_chat_variant(model)
    model.select_variant(variant)

    download_start = time.perf_counter()
    model.download(
        lambda progress: print(
            f"\rDownloading {alias}: {progress:.1f}%",
            end="",
            flush=True,
        )
    )
    print()
    download_time = time.perf_counter() - download_start

    load_start = time.perf_counter()
    model.load()
    load_time = time.perf_counter() - load_start

    results = []
    try:
        client = model.get_chat_client()
        for question in QUESTIONS:
            generation_start = time.perf_counter()
            completion = client.complete_chat(shared_inputs[question]["messages"])
            generation_time = time.perf_counter() - generation_start
            answer = completion.choices[0].message.content or ""
            results.append(
                {
                    "question": question,
                    "answer": answer,
                    "generation_time": generation_time,
                    "citations": CITATION_PATTERN.findall(answer),
                }
            )
    finally:
        model.unload()

    return {
        "alias": alias,
        "variant": variant.id,
        "execution_provider": variant.info.runtime.execution_provider,
        "device_type": str(variant.info.runtime.device_type),
        "download_time": download_time,
        "load_time": load_time,
        "results": results,
    }


def print_shared_retrieval(shared_inputs):
    print("SHARED RETRIEVAL INPUTS")
    for question in QUESTIONS:
        print(f"\nQUESTION: {question}")
        for rank, chunk in enumerate(
            shared_inputs[question]["retrieved_chunks"], start=1
        ):
            print(
                f"{rank}. {chunk['source']} — chunk {chunk['chunk_number']} "
                f"— {chunk['similarity']:.6f}"
            )


def print_model_results(model_result):
    print("\n" + "=" * 72)
    print(f"ALIAS: {model_result['alias']}")
    print(f"VARIANT: {model_result['variant']}")
    print(f"EXECUTION PROVIDER: {model_result['execution_provider']}")
    print(f"DEVICE TYPE: {model_result['device_type']}")
    print(f"DOWNLOAD/CACHE CHECK TIME: {model_result['download_time']:.3f} seconds")
    print(f"MODEL LOAD TIME: {model_result['load_time']:.3f} seconds")

    for result in model_result["results"]:
        print("\nQUESTION:")
        print(result["question"])
        print(f"GENERATION TIME: {result['generation_time']:.3f} seconds")
        print("ANSWER:")
        print(result["answer"])
        print(
            "CITATIONS: "
            + (
                ", ".join(f"[{citation}]" for citation in result["citations"])
                or "None"
            )
        )


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if FoundryLocalManager.instance is None:
        FoundryLocalManager.initialize(Configuration(app_name="foundry_local_rag"))
    manager = FoundryLocalManager.instance

    shared_inputs = retrieve_shared_contexts()
    print_shared_retrieval(shared_inputs)

    aliases = sys.argv[1:] or MODEL_ALIASES
    for alias in aliases:
        try:
            print_model_results(test_model(manager, alias, shared_inputs))
        except Exception as error:
            print("\n" + "=" * 72)
            print(f"ALIAS: {alias}")
            print(f"RUNTIME ERROR: {type(error).__name__}: {error}")

    loaded_models = [model.id for model in manager.catalog.get_loaded_models()]
    print(f"\nLOADED MODELS AFTER COMPARISON: {loaded_models}")


if __name__ == "__main__":
    main()
