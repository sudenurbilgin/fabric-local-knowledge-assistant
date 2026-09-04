import re
import time
from unittest.mock import patch

from fabric_rag import rag as rag_answer
from fabric_rag.rag import RAGSession


QUESTIONS = [
    "What is OneLake?",
    "What is Data Factory in Microsoft Fabric?",
    "Who is the CEO of Microsoft?",
]
CITATION_PATTERN = re.compile(r"\[([1-3])\]")


def measure_question(session, question):
    timings = {"retrieval": 0.0, "generation": 0.0}
    original_retrieval = rag_answer.get_top_chunks
    original_completion = session._chat_client.complete_chat

    def timed_retrieval(*args, **kwargs):
        start = time.perf_counter()
        try:
            return original_retrieval(*args, **kwargs)
        finally:
            timings["retrieval"] = time.perf_counter() - start

    def timed_completion(*args, **kwargs):
        start = time.perf_counter()
        try:
            return original_completion(*args, **kwargs)
        finally:
            timings["generation"] = time.perf_counter() - start

    start = time.perf_counter()
    with patch.object(rag_answer, "get_top_chunks", timed_retrieval), patch.object(
        session._chat_client, "complete_chat", timed_completion
    ):
        result = session.answer(question)
    timings["total"] = time.perf_counter() - start
    timings["other"] = (
        timings["total"] - timings["retrieval"] - timings["generation"]
    )
    return result, timings


def main():
    session_start = time.perf_counter()

    initialization_start = time.perf_counter()
    session = RAGSession()
    initialization_time = time.perf_counter() - initialization_start

    measurements = []
    try:
        for question in QUESTIONS:
            result, timings = measure_question(session, question)
            measurements.append(
                {
                    "question": question,
                    "result": result,
                    **timings,
                }
            )
    finally:
        close_start = time.perf_counter()
        session.close()
        close_time = time.perf_counter() - close_start

    session_time = time.perf_counter() - session_start

    print("PERSISTENT SESSION PERFORMANCE")
    print(f"Initialization + model load: {initialization_time:.3f} seconds")
    print(
        f"{'Question':<43} {'Retrieval':>10} {'Generation':>11} "
        f"{'Other':>9} {'Total':>9}"
    )
    print("-" * 87)
    for measurement in measurements:
        print(
            f"{measurement['question']:<43} "
            f"{measurement['retrieval']:>10.3f} "
            f"{measurement['generation']:>11.3f} "
            f"{measurement['other']:>9.3f} "
            f"{measurement['total']:>9.3f}"
        )

    print(f"Final unload/close: {close_time:.3f} seconds")
    print(f"Full three-question session: {session_time:.3f} seconds")
    print(f"Model: {session.model}")
    print(f"Execution provider: {session.execution_provider}")

    print("\nFUNCTIONAL OUTPUTS")
    for measurement in measurements:
        result = measurement["result"]
        citations = CITATION_PATTERN.findall(result["answer"])
        print(f"\nQUESTION: {measurement['question']}")
        for rank, chunk in enumerate(result["retrieved_chunks"], start=1):
            print(
                f"SOURCE {rank}: {chunk['source']} — chunk "
                f"{chunk['chunk_number']} — {chunk['similarity']:.6f}"
            )
        print("ANSWER:")
        print(result["answer"])
        print(
            "CITATIONS: "
            + (", ".join(f"[{citation}]" for citation in citations) or "None")
        )


if __name__ == "__main__":
    main()
