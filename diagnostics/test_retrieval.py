import json
import subprocess
import sys
from pathlib import Path

from fabric_rag.embeddings import text_preview


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
RESULT_PREFIX = "__RETRIEVAL_RESULTS__="

TEST_CASES = [
    {
        "query": "What is OneLake?",
        "expected_sources": {"onelake-overview.md"},
    },
    {
        "query": "What is Data Factory in Microsoft Fabric?",
        "expected_sources": {"data-factory-overview.md"},
    },
    {
        "query": "What is a lakehouse in Microsoft Fabric?",
        "expected_sources": {"lakehouse-overview.md"},
    },
    {
        "query": "What is Fabric Data Warehouse?",
        "expected_sources": {"data-warehousing.md"},
    },
    {
        "query": "How does Direct Lake work?",
        "expected_sources": {
            "direct-lake-how-it-works.md",
            "direct-lake-overview.md",
        },
    },
    {
        "query": "What services are included in Microsoft Fabric?",
        "expected_sources": {"microsoft-fabric-overview.md"},
    },
]


CHILD_CODE = """
import json
import sys

from fabric_rag.retrieval import get_top_chunks

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

results = get_top_chunks(sys.argv[1], top_k=3)
print("__RETRIEVAL_RESULTS__=" + json.dumps(results, ensure_ascii=False))
"""


def retrieve_in_separate_process(query):
    completed = subprocess.run(
        [sys.executable, "-B", "-c", CHILD_CODE, query],
        cwd=PROJECT_DIRECTORY,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )

    result_line = next(
        (
            line
            for line in completed.stdout.splitlines()
            if line.startswith(RESULT_PREFIX)
        ),
        None,
    )
    if result_line is None:
        raise RuntimeError("The retrieval subprocess did not return structured results.")

    return json.loads(result_line[len(RESULT_PREFIX) :])


def possible_low_value(preview):
    normalized = preview.lower()
    low_value_phrases = (
        "related content",
        "share your feedback",
        "user panel",
        "fabric user panel",
    )
    if any(phrase in normalized for phrase in low_value_phrases):
        return True

    return preview.count("](") >= 3


def print_query_results(query_number, test_case, results):
    print("=" * 60)
    print(f"QUERY {query_number}")
    print("=" * 60)
    print("QUESTION:")
    print(test_case["query"])
    print()

    for rank, result in enumerate(results, start=1):
        preview = text_preview(result["text"], max_chars=300)
        print(f"RANK {rank}")
        print(f"SIMILARITY: {result['similarity']:.6f}")
        print(f"SOURCE: {result['source']}")
        print(f"CHUNK: {result['chunk_number']}")
        print(f"TEXT PREVIEW: {preview}")
        print(
            "POSSIBLE LOW-VALUE CHUNK: "
            f"{'Yes' if possible_low_value(preview) else 'No'}"
        )
        print()

    top_result = results[0]
    expected_match = top_result["source"] in test_case["expected_sources"]
    print(
        "EXPECTED PRIMARY SOURCE: "
        + ", ".join(sorted(test_case["expected_sources"]))
    )
    print(f"TOP-1 SOURCE: {top_result['source']}")
    print(f"TOP-1 EXPECTED SOURCE MATCH: {expected_match}")
    print()

    return expected_match


def print_summary(evaluations):
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(
        f"{'Query':<7} {'Top-1 source':<35} {'Chunk':>5} "
        f"{'Similarity':>11} {'Expected':>9}"
    )
    print("-" * 72)

    for evaluation in evaluations:
        top_result = evaluation["results"][0]
        print(
            f"{evaluation['query_number']:<7} "
            f"{top_result['source']:<35} "
            f"{top_result['chunk_number']:>5} "
            f"{top_result['similarity']:>11.6f} "
            f"{str(evaluation['expected_match']):>9}"
        )

    match_count = sum(evaluation["expected_match"] for evaluation in evaluations)
    print(f"\nExpected-source Top-1 matches: {match_count} / {len(evaluations)}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    evaluations = []
    for query_number, test_case in enumerate(TEST_CASES, start=1):
        results = retrieve_in_separate_process(test_case["query"])
        expected_match = print_query_results(
            query_number, test_case, results
        )
        evaluations.append(
            {
                "query_number": query_number,
                "test_case": test_case,
                "results": results,
                "expected_match": expected_match,
            }
        )

    print_summary(evaluations)


if __name__ == "__main__":
    main()
