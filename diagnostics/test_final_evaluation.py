import re
import sys
import time

from foundry_local_sdk import FoundryLocalManager

from fabric_rag.rag import RAGSession


TEST_CASES = [
    ("SUPPORTED", "What is Microsoft Fabric?"),
    ("SUPPORTED", "What is OneLake?"),
    ("SUPPORTED", "What is Data Factory in Microsoft Fabric?"),
    ("SUPPORTED", "How does Direct Lake work?"),
    ("UNSUPPORTED", "Who is the CEO of Microsoft?"),
    ("UNSUPPORTED", "What is the weather in Istanbul today?"),
    ("GENERAL", "Tell me about Microsoft Fabric."),
]
CITATION_PATTERN = re.compile(r"\[([1-3])\]")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    evaluation_start = time.perf_counter()
    session = RAGSession()
    try:
        for test_number, (category, question) in enumerate(TEST_CASES, start=1):
            question_start = time.perf_counter()
            result = session.answer(question)
            question_time = time.perf_counter() - question_start
            citations = CITATION_PATTERN.findall(result["answer"])

            print("=" * 72)
            print(f"TEST {test_number}")
            print(f"CATEGORY: {category}")
            print(f"QUESTION: {question}")
            print("TOP-3 SOURCES:")
            for rank, chunk in enumerate(result["retrieved_chunks"], start=1):
                print(
                    f"{rank}. {chunk['source']} — chunk {chunk['chunk_number']} "
                    f"— {chunk['similarity']:.6f}"
                )
            print("ANSWER:")
            print(result["answer"])
            print(
                "CITATION LABELS USED: "
                + (", ".join(f"[{citation}]" for citation in citations) or "None")
            )
            print(f"TOTAL QUESTION TIME: {question_time:.3f} seconds")
            print()
    finally:
        session.close()

    loaded_models = [
        model.id
        for model in FoundryLocalManager.instance.catalog.get_loaded_models()
    ]
    print("=" * 72)
    print(f"MODEL: {session.model}")
    print(f"EXECUTION PROVIDER: {session.execution_provider}")
    print(f"FULL EVALUATION TIME: {time.perf_counter() - evaluation_start:.3f} seconds")
    print(f"LOADED MODELS AFTER CLEANUP: {loaded_models}")


if __name__ == "__main__":
    main()
