import re
import sys
import time

from foundry_local_sdk import FoundryLocalManager

from fabric_rag.rag import RAGSession


CITATION_PATTERN = re.compile(r"\[([1-3])\]")
MARKDOWN_URL_PATTERN = re.compile(r"\[[^\]]+\]\([^\)]+\)|https?://|www\.")
NAVIGATION_PATTERN = re.compile(
    r"(?:for more information|to learn more|related content)",
    re.IGNORECASE,
)
REFUSAL_PATTERN = re.compile(
    r"(?:not enough information|does not (?:contain|include|provide)|"
    r"cannot (?:answer|determine)|unable to (?:answer|determine))",
    re.IGNORECASE,
)


SUPPORTED_CASES = [
    {
        "topic": "Microsoft Fabric overview",
        "question": "What is Microsoft Fabric and what kinds of data workflows does it support?",
        "context_terms": ("Microsoft Fabric", "data ingestion", "reporting"),
    },
    {
        "topic": "OneLake",
        "question": "What is OneLake, and how does it help avoid duplicate copies of data?",
        "context_terms": ("OneLake", "single", "duplication"),
    },
    {
        "topic": "Lakehouse",
        "question": "What is a lakehouse in Microsoft Fabric?",
        "context_terms": ("lakehouse", "data lake", "warehouse"),
    },
    {
        "topic": "Lakehouse SQL endpoint",
        "question": "What can you do with the SQL analytics endpoint of a Fabric lakehouse?",
        "context_terms": ("SQL analytics endpoint", "T-SQL"),
    },
    {
        "topic": "Fabric Data Warehouse",
        "question": "What is Fabric Data Warehouse?",
        "context_terms": ("Fabric Data Warehouse", "T-SQL"),
    },
    {
        "topic": "Warehouse or lakehouse",
        "question": "When should you choose a Fabric warehouse instead of a lakehouse?",
        "context_terms": ("Warehouse", "lakehouse"),
    },
    {
        "topic": "Data Factory",
        "question": "What is Data Factory in Microsoft Fabric, and what data integration capabilities does it provide?",
        "context_terms": ("Data Factory", "data integration"),
    },
    {
        "topic": "Direct Lake overview",
        "question": "What is Direct Lake and what are its main benefits?",
        "context_terms": ("Direct Lake", "VertiPaq"),
    },
    {
        "topic": "Direct Lake transcoding",
        "question": "How does Direct Lake column loading, or transcoding, work?",
        "context_terms": ("transcoding", "columns", "memory"),
    },
    {
        "topic": "Direct Lake framing",
        "question": "What is framing in Direct Lake, and how does it affect the data seen by queries?",
        "context_terms": ("Framing", "Delta"),
    },
    {
        "topic": "DirectQuery fallback",
        "question": "When can Direct Lake fall back to DirectQuery, and why should fallback be avoided?",
        "context_terms": ("DirectQuery fallback", "slower"),
    },
    {
        "topic": "Data lifecycle",
        "question": "What are the six stages of the end-to-end data lifecycle in Microsoft Fabric?",
        "context_terms": ("Get data", "Store data", "External integration"),
    },
]


UNSUPPORTED_CASES = [
    {
        "topic": "Microsoft leadership",
        "question": "Who is the CEO of Microsoft?",
        "forbidden_answers": ("Satya Nadella",),
    },
    {
        "topic": "Current weather",
        "question": "What is the current weather in Istanbul?",
        "forbidden_answers": (
            "sunny",
            "cloudy",
            "raining",
            "degrees",
            "°C",
            "°F",
        ),
    },
    {
        "topic": "Unrelated geography",
        "question": "What is the capital of Japan?",
        "forbidden_answers": ("Tokyo",),
    },
]


def normalize(text):
    return " ".join(text.casefold().split())


def evaluate_supported(case, result):
    answer = result["answer"] or ""
    chunks = result["retrieved_chunks"]
    context = normalize("\n".join(chunk["text"] for chunk in chunks))
    citations = CITATION_PATTERN.findall(answer)
    valid_ranks = {str(rank) for rank in range(1, len(chunks) + 1)}

    checks = {
        "context_sufficient": all(
            normalize(term) in context for term in case["context_terms"]
        ),
        "answer_present": bool(answer.strip()),
        "not_refused": REFUSAL_PATTERN.search(answer) is None,
        "citations_present": bool(citations),
        "citation_ranks_valid": set(citations).issubset(valid_ranks),
        "no_markdown_urls": MARKDOWN_URL_PATTERN.search(answer) is None,
        "no_navigation_artifacts": NAVIGATION_PATTERN.search(answer) is None,
    }
    return checks, all(checks.values())


def evaluate_unsupported(case, result):
    answer = result["answer"] or ""
    answer_folded = answer.casefold()
    checks = {
        "refused": REFUSAL_PATTERN.search(answer) is not None,
        "requested_fact_not_given": all(
            forbidden.casefold() not in answer_folded
            for forbidden in case["forbidden_answers"]
        ),
        "no_citations_presented_as_answer": not CITATION_PATTERN.findall(answer),
    }
    return checks, all(checks.values())


def print_case_evidence(case, supported, result, elapsed, checks, passed):
    print("\n" + "=" * 80)
    print(f"TOPIC: {case['topic']}")
    print(f"TYPE: {'SUPPORTED' if supported else 'UNSUPPORTED'}")
    print(f"QUESTION: {case['question']}")
    print(f"TOTAL ANSWER TIME: {elapsed:.3f} seconds")
    print("\nTOP-3 RETRIEVED CONTEXT")
    for rank, chunk in enumerate(result["retrieved_chunks"], start=1):
        print("\n" + "-" * 80)
        print(
            f"RANK {rank} | SOURCE: {chunk['source']} | "
            f"CHUNK: {chunk['chunk_number']} | "
            f"SIMILARITY: {chunk['similarity']:.6f}"
        )
        preview = " ".join(chunk["text"].split())
        if len(preview) > 500:
            preview = preview[:497].rstrip() + "..."
        print(f"TEXT PREVIEW: {preview}")

    answer = result["answer"] or ""
    citations = CITATION_PATTERN.findall(answer)
    print("\nANSWER")
    print(answer)
    print(
        "\nCITATIONS USED: "
        + (", ".join(f"[{label}]" for label in citations) or "None")
    )
    print("ACCEPTANCE CHECKS:")
    for name, value in checks.items():
        print(f"- {name}: {'PASS' if value else 'FAIL'}")
    print(f"AUTOMATED RESULT: {'PASS' if passed else 'FAIL'}")


def print_final_summary(records, session_completed, loaded_models):
    supported_records = [record for record in records if record["supported"]]
    unsupported_records = [record for record in records if not record["supported"]]
    topics = ", ".join(record["topic"] for record in supported_records)

    print("\n" + "=" * 80)
    print("FINAL CORPUS COVERAGE EVALUATION")
    print("=" * 80)
    print("Topic | Supported/Unsupported | Result")
    print("-" * 80)
    for record in records:
        case_type = "Supported" if record["supported"] else "Unsupported"
        result = "PASS" if record["passed"] else "FAIL"
        print(f"{record['topic']} | {case_type} | {result}")

    supported_passes = sum(record["passed"] for record in supported_records)
    unsupported_passes = sum(record["passed"] for record in unsupported_records)
    print(
        f"\nSupported questions passed: {supported_passes} / "
        f"{len(supported_records)}"
    )
    print(
        f"Unsupported questions passed: {unsupported_passes} / "
        f"{len(unsupported_records)}"
    )
    print(f"Topics represented: {topics}")
    print(f"Session completed: {'YES' if session_completed else 'NO'}")
    print(f"Loaded models after cleanup: {loaded_models}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    records = []
    session_completed = False
    session = None
    try:
        session = RAGSession()
        print(f"PRODUCTION MODEL: {session.model}")
        print(f"EXECUTION PROVIDER: {session.execution_provider}")

        for supported, cases in (
            (True, SUPPORTED_CASES),
            (False, UNSUPPORTED_CASES),
        ):
            for case in cases:
                start = time.perf_counter()
                result = session.answer(case["question"])
                elapsed = time.perf_counter() - start
                if supported:
                    checks, passed = evaluate_supported(case, result)
                else:
                    checks, passed = evaluate_unsupported(case, result)

                print_case_evidence(
                    case,
                    supported,
                    result,
                    elapsed,
                    checks,
                    passed,
                )
                records.append(
                    {
                        "topic": case["topic"],
                        "supported": supported,
                        "passed": passed,
                    }
                )
        session_completed = True
    finally:
        if session is not None:
            session.close()

    manager = FoundryLocalManager.instance
    loaded_models = (
        [model.id for model in manager.catalog.get_loaded_models()]
        if manager is not None
        else []
    )
    print_final_summary(records, session_completed, loaded_models)


if __name__ == "__main__":
    main()
