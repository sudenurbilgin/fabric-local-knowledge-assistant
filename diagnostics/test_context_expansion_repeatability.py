import re
import sys
import time

from foundry_local_sdk import FoundryLocalManager

from fabric_rag.rag import RAGSession, build_context, build_messages
from fabric_rag.retrieval import get_top_chunks
from .test_retrieval_context_expansion import prepare_strategies
from .retrieval_performance_baseline import (
    BASELINE_CHUNK_COUNT,
    require_baseline_corpus,
)


EXPECTED_MODEL = "Phi-4-mini-instruct-generic-cpu:5"
EXPECTED_PROVIDER = "CPUExecutionProvider"
REPETITIONS = 3
LIFECYCLE_QUESTION = (
    "What are the six stages of the end-to-end data lifecycle in Microsoft Fabric?"
)
DIRECT_LAKE_QUESTION = "How does Direct Lake work?"
REQUIRED_LIFECYCLE_STAGES = [
    "Get data",
    "Store data",
    "Prepare and transform",
    "Analyze and train",
    "Track and visualize",
    "External integration",
]
CITATION_PATTERN = re.compile(r"\[([0-9]+)\]")
MARKDOWN_URL_PATTERN = re.compile(r"\[[^\]]+\]\([^\)]+\)|https?://|www\.")
REFUSAL_PATTERN = re.compile(
    r"(?:not enough information|insufficient (?:information|context)|"
    r"does not (?:contain|include|provide)|cannot (?:answer|provide|determine)|"
    r"unable to (?:answer|provide|determine))",
    re.IGNORECASE,
)
LIST_LABEL_PATTERN = re.compile(
    r"(?m)^\s*(?:\d+\.|[-*]|\[[0-9]+\]\s*[-*]?)\s*"
    r"\*\*([^*:]+)(?::[^*]*)?\*\*"
)


def normalize(text):
    return " ".join(text.casefold().split())


def fixed_inputs(topic, question):
    # This is the only semantic retrieval call for this question.
    ranked = get_top_chunks(question, top_k=BASELINE_CHUNK_COUNT)
    strategies = prepare_strategies(ranked)
    baseline_chunks = strategies["A. Baseline Top-3 only"]["chunks"]
    expanded_strategy = strategies[
        "C. Top-3 plus most relevant rank-1 neighbor"
    ]
    expanded_chunks = expanded_strategy["chunks"]

    baseline_context = build_context(baseline_chunks)
    expanded_context = build_context(expanded_chunks)
    return {
        "topic": topic,
        "question": question,
        "baseline_chunks": baseline_chunks,
        "expanded_chunks": expanded_chunks,
        "neighbor": expanded_strategy["neighbors"][0]
        if expanded_strategy["neighbors"]
        else None,
        "baseline_context": baseline_context,
        "expanded_context": expanded_context,
        # Build each payload once and reuse it unchanged for all repetitions.
        "baseline_messages": build_messages(question, baseline_context),
        "expanded_messages": build_messages(question, expanded_context),
    }


def print_fixed_inputs(case):
    print("\n" + "=" * 80)
    print(f"FIXED RETRIEVAL INPUTS: {case['topic'].upper()}")
    print("=" * 80)
    print(f"Question: {case['question']}")
    print("\nSemantic Top-3:")
    for rank, chunk in enumerate(case["baseline_chunks"], start=1):
        preview = " ".join(chunk["text"].split())
        if len(preview) > 600:
            preview = preview[:597].rstrip() + "..."
        print(
            f"Rank {rank}: {chunk['source']} chunk {chunk['chunk_number']} | "
            f"similarity {chunk['similarity']:.6f}"
        )
        print(f"Preview: {preview}")

    neighbor = case["neighbor"]
    if neighbor is None:
        print("\nFocused rank-1 neighbor added: none")
    else:
        preview = " ".join(neighbor["text"].split())
        if len(preview) > 600:
            preview = preview[:597].rstrip() + "..."
        print(
            "\nFocused rank-1 neighbor added under label [1]: "
            f"{neighbor['source']} chunk {neighbor['chunk_number']} | "
            f"similarity {neighbor['similarity']:.6f}"
        )
        print(f"Preview: {preview}")
    print(f"Baseline context characters: {len(case['baseline_context'])}")
    print(f"Expanded context characters: {len(case['expanded_context'])}")


def generate(session, messages):
    started = time.perf_counter()
    completion = session._chat_client.complete_chat(messages)
    elapsed = time.perf_counter() - started
    return completion.choices[0].message.content or "", elapsed


def citation_status(answer):
    citations = CITATION_PATTERN.findall(answer)
    return citations, bool(citations), set(citations).issubset({"1", "2", "3"})


def has_mixed_refusal(answer, substantive):
    return substantive and REFUSAL_PATTERN.search(answer) is not None


def lifecycle_checks(answer):
    normalized_answer = normalize(answer)
    stages_present = [
        stage
        for stage in REQUIRED_LIFECYCLE_STAGES
        if normalize(stage) in normalized_answer
    ]
    all_six = len(stages_present) == len(REQUIRED_LIFECYCLE_STAGES)
    citations, citations_present, valid_citations = citation_status(answer)
    listed_labels = [normalize(label) for label in LIST_LABEL_PATTERN.findall(answer)]
    allowed_labels = {normalize(stage) for stage in REQUIRED_LIFECYCLE_STAGES}
    outside_labels = [label for label in listed_labels if label not in allowed_labels]
    substantive = len(normalized_answer) >= 100 or len(stages_present) >= 2
    mixed_refusal = has_mixed_refusal(answer, substantive)
    no_urls = MARKDOWN_URL_PATTERN.search(answer) is None
    passed = all(
        (
            all_six,
            citations_present,
            valid_citations,
            not outside_labels,
            not mixed_refusal,
            no_urls,
        )
    )
    return {
        "all_six_stages": all_six,
        "stages_present": stages_present,
        "citations": citations,
        "citations_present": citations_present,
        "valid_citations": valid_citations,
        "outside_stage_labels": outside_labels,
        "mixed_refusal": mixed_refusal,
        "no_markdown_urls": no_urls,
        "passed": passed,
    }


def direct_lake_checks(answer):
    normalized_answer = normalize(answer)
    relevant = "direct lake" in normalized_answer
    substantive = relevant and len(normalized_answer) >= 120
    refused = REFUSAL_PATTERN.search(answer) is not None
    mixed_refusal = has_mixed_refusal(answer, substantive)
    citations, citations_present, valid_citations = citation_status(answer)
    no_urls = MARKDOWN_URL_PATTERN.search(answer) is None
    passed = all(
        (
            substantive,
            not refused,
            citations_present,
            valid_citations,
            not mixed_refusal,
            no_urls,
        )
    )
    return {
        "grounded": substantive and citations_present and valid_citations,
        "substantive_relevant": substantive,
        "supported_answer": not refused,
        "citations": citations,
        "citations_present": citations_present,
        "valid_citations": valid_citations,
        "mixed_refusal": mixed_refusal,
        "no_markdown_urls": no_urls,
        "passed": passed,
    }


def print_lifecycle_run(condition, run_number, answer, elapsed, checks):
    print("\n" + "-" * 80)
    print(f"Lifecycle | {condition} | Run {run_number}")
    print(f"Generation time: {elapsed:.3f} seconds")
    print("Answer:")
    print(answer)
    print("Checks:")
    print(f"All six stages: {'YES' if checks['all_six_stages'] else 'NO'}")
    print(f"Stages found: {', '.join(checks['stages_present']) or 'none'}")
    print(f"Citations present: {'YES' if checks['citations_present'] else 'NO'}")
    print(f"Valid citations: {'YES' if checks['valid_citations'] else 'NO'}")
    print(
        "Unsupported stage substitutions: "
        f"{', '.join(checks['outside_stage_labels']) or 'none'}"
    )
    print(f"Mixed refusal: {'YES' if checks['mixed_refusal'] else 'NO'}")
    print(f"Markdown URLs: {'NO' if checks['no_markdown_urls'] else 'YES'}")
    print(f"Result: {'PASS' if checks['passed'] else 'FAIL'}")


def print_direct_lake_run(condition, run_number, answer, elapsed, checks):
    print("\n" + "-" * 80)
    print(f"Direct Lake | {condition} | Run {run_number}")
    print(f"Generation time: {elapsed:.3f} seconds")
    print("Answer:")
    print(answer)
    print("Checks:")
    print(f"Grounded: {'YES' if checks['grounded'] else 'NO'}")
    print(
        "Substantive and relevant: "
        f"{'YES' if checks['substantive_relevant'] else 'NO'}"
    )
    print(f"Supported answer: {'YES' if checks['supported_answer'] else 'NO'}")
    print(f"Citations present: {'YES' if checks['citations_present'] else 'NO'}")
    print(f"Valid citations: {'YES' if checks['valid_citations'] else 'NO'}")
    print(f"Mixed refusal: {'YES' if checks['mixed_refusal'] else 'NO'}")
    print(f"Markdown URLs: {'NO' if checks['no_markdown_urls'] else 'YES'}")
    print(f"Result: {'PASS' if checks['passed'] else 'FAIL'}")


def run_condition(session, case, condition):
    messages = case[f"{condition.casefold()}_messages"]
    records = []
    for run_number in range(1, REPETITIONS + 1):
        answer, elapsed = generate(session, messages)
        if case["topic"] == "Lifecycle":
            checks = lifecycle_checks(answer)
            print_lifecycle_run(condition, run_number, answer, elapsed, checks)
        else:
            checks = direct_lake_checks(answer)
            print_direct_lake_run(condition, run_number, answer, elapsed, checks)
        records.append(
            {
                "run": run_number,
                "answer": answer,
                "elapsed": elapsed,
                "checks": checks,
                "passed": checks["passed"],
            }
        )
    return records


def average_time(records):
    return sum(record["elapsed"] for record in records) / len(records)


def classify(results):
    lifecycle_baseline = sum(
        record["passed"] for record in results["Lifecycle"]["Baseline"]
    )
    lifecycle_expanded = sum(
        record["passed"] for record in results["Lifecycle"]["Expanded"]
    )
    direct_expanded = sum(
        record["passed"] for record in results["Direct Lake"]["Expanded"]
    )

    if lifecycle_expanded == 3 and lifecycle_baseline < 3 and direct_expanded == 3:
        return (
            "A. STRONG IMPROVEMENT",
            "Focused neighbor expansion is repeatably beneficial and is a candidate "
            "for minimal production integration.",
        )
    if lifecycle_expanded == 3 and lifecycle_baseline < 3 and direct_expanded < 3:
        return (
            "B. PROMISING BUT MIXED",
            "Do not integrate blanket neighbor expansion into production yet. "
            "Investigate a conditional/adaptive expansion strategy.",
        )
    return (
        "C. NO RELIABLE IMPROVEMENT",
        "Keep production unchanged.",
    )


def print_summary(results, loaded_models):
    classification, recommendation = classify(results)
    print("\n" + "=" * 80)
    print("CONTEXT EXPANSION REPEATABILITY SUMMARY")
    print("=" * 80)
    for topic, success_label in (
        ("Lifecycle", "Successful complete answers"),
        ("Direct Lake", "Successful grounded answers"),
    ):
        print(f"\n{topic.upper()}")
        for condition in ("Baseline", "Expanded"):
            records = results[topic][condition]
            print(f"{condition}:")
            for record in records:
                print(
                    f"Run {record['run']}: "
                    f"{'PASS' if record['passed'] else 'FAIL'}"
                )
            print(
                f"{success_label}: "
                f"{sum(record['passed'] for record in records)}/3"
            )

    print("\nAVERAGE GENERATION TIMES")
    print(
        "Lifecycle baseline: "
        f"{average_time(results['Lifecycle']['Baseline']):.3f} s"
    )
    print(
        "Lifecycle expanded: "
        f"{average_time(results['Lifecycle']['Expanded']):.3f} s"
    )
    print(
        "Direct Lake baseline: "
        f"{average_time(results['Direct Lake']['Baseline']):.3f} s"
    )
    print(
        "Direct Lake expanded: "
        f"{average_time(results['Direct Lake']['Expanded']):.3f} s"
    )
    print("\nFINAL INTERPRETATION")
    print(classification)
    print(recommendation)
    print(f"\nLoaded models after cleanup: {loaded_models}")


def main():
    require_baseline_corpus()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # Both retrievals happen before generation; no retrieval occurs in the loops.
    cases = [
        fixed_inputs("Lifecycle", LIFECYCLE_QUESTION),
        fixed_inputs("Direct Lake", DIRECT_LAKE_QUESTION),
    ]
    for case in cases:
        print_fixed_inputs(case)

    results = {
        case["topic"]: {"Baseline": [], "Expanded": []} for case in cases
    }
    session = None
    try:
        session = RAGSession()
        if session.model != EXPECTED_MODEL:
            raise RuntimeError(
                f"Expected model {EXPECTED_MODEL}, but selected {session.model}."
            )
        if session.execution_provider != EXPECTED_PROVIDER:
            raise RuntimeError(
                f"Expected provider {EXPECTED_PROVIDER}, but selected "
                f"{session.execution_provider}."
            )
        print(f"\nMODEL: {session.model}")
        print(f"PROVIDER: {session.execution_provider}")

        for case in cases:
            for condition in ("Baseline", "Expanded"):
                results[case["topic"]][condition] = run_condition(
                    session, case, condition
                )
    finally:
        if session is not None:
            session.close()

    manager = FoundryLocalManager.instance
    loaded_models = (
        [model.id for model in manager.catalog.get_loaded_models()]
        if manager is not None
        else []
    )
    print_summary(results, loaded_models)


if __name__ == "__main__":
    main()
