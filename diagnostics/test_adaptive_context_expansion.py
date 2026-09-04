import re
import sys
import time

from foundry_local_sdk import FoundryLocalManager

from fabric_rag.rag import RAGSession, build_context, build_messages
from fabric_rag.retrieval import get_top_chunks
from .test_final_coverage import (
    SUPPORTED_CASES,
    UNSUPPORTED_CASES,
    evaluate_supported,
    evaluate_unsupported,
)
from .test_retrieval_context_expansion import prepare_strategies
from .retrieval_performance_baseline import (
    BASELINE_CHUNK_COUNT,
    require_baseline_corpus,
)


EXPECTED_MODEL = "Phi-4-mini-instruct-generic-cpu:5"
EXPECTED_PROVIDER = "CPUExecutionProvider"
MINIMUM_SAFE_RANGE_GAP = 0.10
INSUFFICIENT_CONTEXT_PATTERN = re.compile(
    r"(?:not enough information|insufficient (?:information|context)|"
    r"does not (?:contain|include|provide)|do not have enough information|"
    r"cannot (?:answer|provide|determine)|unable to (?:answer|provide|determine)|"
    r"context (?:is|was) incomplete|remaining .* (?:not|missing))",
    re.IGNORECASE,
)
CITATION_PATTERN = re.compile(r"\[([1-3])\]")
MARKDOWN_URL_PATTERN = re.compile(r"\[[^\]]+\]\([^\)]+\)|https?://|www\.")
LIFECYCLE_STAGES = [
    "Get data",
    "Store data",
    "Prepare and transform",
    "Analyze and train",
    "Track and visualize",
    "External integration",
]
LIFECYCLE_QUESTION = (
    "What are the six stages of the end-to-end data lifecycle in Microsoft Fabric?"
)


INITIAL_CASES = [
    {
        "topic": "Lifecycle",
        "question": LIFECYCLE_QUESTION,
        "supported": True,
        "context_terms": ("Get data", "Store data", "External integration"),
        "expected_retry": True,
        "lifecycle": True,
    },
    {
        "topic": "Direct Lake",
        "question": "How does Direct Lake work?",
        "supported": True,
        "context_terms": ("Direct Lake", "VertiPaq"),
        "expected_retry": False,
    },
    {
        "topic": "OneLake",
        "question": "What is OneLake?",
        "supported": True,
        "context_terms": ("OneLake", "unified data lake"),
        "expected_retry": False,
    },
    {
        "topic": "Lakehouse",
        "question": "What is a lakehouse in Microsoft Fabric?",
        "supported": True,
        "context_terms": ("lakehouse", "data lake", "warehouse"),
        "expected_retry": False,
    },
    {
        "topic": "Data Factory",
        "question": "What is Data Factory in Microsoft Fabric?",
        "supported": True,
        "context_terms": ("Data Factory", "data integration"),
        "expected_retry": False,
    },
    {
        "topic": "Microsoft leadership",
        "question": "Who is the CEO of Microsoft?",
        "supported": False,
        "forbidden_answers": ("Satya Nadella",),
        "expected_retry": False,
    },
    {
        "topic": "Current weather",
        "question": "What is the current weather in Istanbul?",
        "supported": False,
        "forbidden_answers": (
            "sunny",
            "cloudy",
            "raining",
            "degrees",
            "°C",
            "°F",
        ),
        "expected_retry": False,
    },
    {
        "topic": "Unrelated geography",
        "question": "What is the capital of Japan?",
        "supported": False,
        "forbidden_answers": ("Tokyo",),
        "expected_retry": False,
    },
]


def normalize(text):
    return " ".join(text.casefold().split())


def retrieve_once(question, retrieval_cache):
    if question not in retrieval_cache:
        ranked = get_top_chunks(question, top_k=BASELINE_CHUNK_COUNT)
        strategies = prepare_strategies(ranked)
        retrieval_cache[question] = {
            "ranked": ranked,
            "baseline_chunks": strategies["A. Baseline Top-3 only"]["chunks"],
            "expanded_chunks": strategies[
                "C. Top-3 plus most relevant rank-1 neighbor"
            ]["chunks"],
            "neighbor": strategies[
                "C. Top-3 plus most relevant rank-1 neighbor"
            ]["neighbors"],
        }
    return retrieval_cache[question]


def collect_coverage_retrievals(retrieval_cache):
    rows = []
    for supported, cases in (
        (True, SUPPORTED_CASES),
        (False, UNSUPPORTED_CASES),
    ):
        for case in cases:
            retrieval = retrieve_once(case["question"], retrieval_cache)
            top_three = retrieval["baseline_chunks"]
            rows.append(
                {
                    "topic": case["topic"],
                    "question": case["question"],
                    "supported": supported,
                    "top1": top_three[0]["similarity"],
                    "top2": top_three[1]["similarity"],
                    "top3": top_three[2]["similarity"],
                }
            )
    return rows


def print_similarity_table(rows):
    print("\n" + "=" * 100)
    print("FINAL COVERAGE RETRIEVAL SIMILARITIES")
    print("=" * 100)
    print("Topic | Category | Top-1 | Top-2 | Top-3")
    print("-" * 100)
    for row in rows:
        category = "Supported" if row["supported"] else "Unsupported"
        print(
            f"{row['topic']} | {category} | {row['top1']:.6f} | "
            f"{row['top2']:.6f} | {row['top3']:.6f}"
        )


def derive_threshold(rows):
    supported_top1 = [row["top1"] for row in rows if row["supported"]]
    unsupported_top1 = [row["top1"] for row in rows if not row["supported"]]
    supported_minimum = min(supported_top1)
    unsupported_maximum = max(unsupported_top1)
    range_gap = supported_minimum - unsupported_maximum
    threshold = (supported_minimum + unsupported_maximum) / 2
    safety_margin = range_gap / 2
    defensible = range_gap >= MINIMUM_SAFE_RANGE_GAP
    return {
        "supported_minimum": supported_minimum,
        "unsupported_maximum": unsupported_maximum,
        "range_gap": range_gap,
        "threshold": threshold,
        "safety_margin": safety_margin,
        "defensible": defensible,
    }


def print_threshold_analysis(analysis):
    print("\nTHRESHOLD ANALYSIS")
    print(f"Lowest supported Top-1: {analysis['supported_minimum']:.6f}")
    print(f"Highest unsupported Top-1: {analysis['unsupported_maximum']:.6f}")
    print(f"Observed range gap: {analysis['range_gap']:.6f}")
    print(
        "Midpoint threshold with equal observed safety margins: "
        f"{analysis['threshold']:.6f}"
    )
    print(f"Margin on each side: {analysis['safety_margin']:.6f}")
    print(
        "Threshold defensible for this corpus: "
        f"{'YES' if analysis['defensible'] else 'NO'}"
    )


def print_fixed_retrieval(topic, retrieval):
    print(f"\nFIXED RETRIEVAL: {topic}")
    for rank, chunk in enumerate(retrieval["baseline_chunks"], start=1):
        print(
            f"Rank {rank}: {chunk['source']} chunk {chunk['chunk_number']} | "
            f"{chunk['similarity']:.6f}"
        )
    if retrieval["neighbor"]:
        neighbor = retrieval["neighbor"][0]
        print(
            "Focused neighbor candidate: "
            f"{neighbor['source']} chunk {neighbor['chunk_number']} | "
            f"{neighbor['similarity']:.6f}"
        )
    else:
        print("Focused neighbor candidate: none")


def generate(session, messages):
    started = time.perf_counter()
    completion = session._chat_client.complete_chat(messages)
    return completion.choices[0].message.content or "", time.perf_counter() - started


def adaptive_answer(session, question, retrieval, threshold):
    baseline_context = build_context(retrieval["baseline_chunks"])
    baseline_messages = build_messages(question, baseline_context)
    baseline_answer, baseline_time = generate(session, baseline_messages)

    insufficient_signal = INSUFFICIENT_CONTEXT_PATTERN.search(baseline_answer) is not None
    high_confidence = retrieval["baseline_chunks"][0]["similarity"] >= threshold
    neighbor_available = bool(retrieval["neighbor"])
    retry_triggered = insufficient_signal and high_confidence and neighbor_available

    final_answer = baseline_answer
    final_chunks = retrieval["baseline_chunks"]
    retry_time = 0.0
    if retry_triggered:
        expanded_context = build_context(retrieval["expanded_chunks"])
        expanded_messages = build_messages(question, expanded_context)
        final_answer, retry_time = generate(session, expanded_messages)
        final_chunks = retrieval["expanded_chunks"]

    return {
        "question": question,
        "top1": retrieval["baseline_chunks"][0]["similarity"],
        "baseline_answer": baseline_answer,
        "baseline_time": baseline_time,
        "baseline_insufficient": insufficient_signal,
        "high_confidence": high_confidence,
        "neighbor_available": neighbor_available,
        "retry_triggered": retry_triggered,
        "retry_time": retry_time,
        "total_time": baseline_time + retry_time,
        "final_answer": final_answer,
        "final_chunks": final_chunks,
        "final_condition": "EXPANDED" if retry_triggered else "BASELINE",
    }


def evaluate_case(case, adaptive_result):
    result = {
        "answer": adaptive_result["final_answer"],
        "retrieved_chunks": adaptive_result["final_chunks"],
    }
    if case["supported"]:
        checks, passed = evaluate_supported(case, result)
        if case.get("lifecycle"):
            normalized_answer = normalize(adaptive_result["final_answer"])
            all_six = all(normalize(stage) in normalized_answer for stage in LIFECYCLE_STAGES)
            checks["all_six_lifecycle_stages"] = all_six
            passed = passed and all_six
    else:
        checks, passed = evaluate_unsupported(case, result)
    return checks, passed


def print_adaptive_case(label, case, adaptive_result, checks, passed, reused=False):
    print("\n" + "=" * 100)
    print(f"{label}: {case['topic']}")
    print("=" * 100)
    print(f"Question: {case['question']}")
    print(f"Top-1 similarity: {adaptive_result['top1']:.6f}")
    print(
        "Baseline answer status: "
        f"{'INSUFFICIENT CONTEXT' if adaptive_result['baseline_insufficient'] else 'ANSWERED'}"
    )
    print(f"High-confidence gate: {'YES' if adaptive_result['high_confidence'] else 'NO'}")
    print(f"Neighbor available: {'YES' if adaptive_result['neighbor_available'] else 'NO'}")
    print(f"Retry triggered: {'YES' if adaptive_result['retry_triggered'] else 'NO'}")
    print(f"Final context: {adaptive_result['final_condition']}")
    print(f"Baseline generation time: {adaptive_result['baseline_time']:.3f} s")
    print(f"Additional retry cost: {adaptive_result['retry_time']:.3f} s")
    print(f"Total time: {adaptive_result['total_time']:.3f} s")
    print(f"Reused from initial phase: {'YES' if reused else 'NO'}")
    print("\nBASELINE ANSWER")
    print(adaptive_result["baseline_answer"])
    if adaptive_result["retry_triggered"]:
        print("\nADAPTIVE FINAL ANSWER")
        print(adaptive_result["final_answer"])
    print("\nACCEPTANCE CHECKS")
    for name, value in checks.items():
        print(f"- {name}: {'PASS' if value else 'FAIL'}")
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")


def initial_cases_passed(records):
    for record in records:
        expected_retry = record["case"]["expected_retry"]
        if not record["passed"] or record["adaptive"]["retry_triggered"] != expected_retry:
            return False
    return True


def evaluate_full_coverage(session, retrieval_cache, threshold, initial_by_question):
    records = []
    for supported, cases in (
        (True, SUPPORTED_CASES),
        (False, UNSUPPORTED_CASES),
    ):
        for source_case in cases:
            case = dict(source_case)
            case["supported"] = supported
            case["lifecycle"] = case["question"] == LIFECYCLE_QUESTION
            if case["question"] in initial_by_question:
                adaptive_result = initial_by_question[case["question"]]["adaptive"]
                reused = True
            else:
                retrieval = retrieve_once(case["question"], retrieval_cache)
                adaptive_result = adaptive_answer(
                    session, case["question"], retrieval, threshold
                )
                reused = False
            checks, passed = evaluate_case(case, adaptive_result)
            print_adaptive_case(
                "FULL COVERAGE",
                case,
                adaptive_result,
                checks,
                passed,
                reused=reused,
            )
            records.append(
                {
                    "case": case,
                    "adaptive": adaptive_result,
                    "checks": checks,
                    "passed": passed,
                }
            )
    return records


def classification(records):
    supported = [record for record in records if record["case"]["supported"]]
    unsupported = [record for record in records if not record["case"]["supported"]]
    lifecycle = next(
        record for record in supported if record["case"]["question"] == LIFECYCLE_QUESTION
    )
    lifecycle_normalized = normalize(lifecycle["adaptive"]["final_answer"])
    lifecycle_complete = all(
        normalize(stage) in lifecycle_normalized for stage in LIFECYCLE_STAGES
    )
    unsupported_retries = sum(
        record["adaptive"]["retry_triggered"] for record in unsupported
    )
    supported_passes = sum(record["passed"] for record in supported)
    unsupported_passes = sum(record["passed"] for record in unsupported)
    if (
        lifecycle["adaptive"]["retry_triggered"]
        and lifecycle_complete
        and supported_passes == len(supported)
        and unsupported_passes == len(unsupported)
        and unsupported_retries == 0
    ):
        return (
            "A. READY FOR PRODUCTION EXPERIMENT",
            "Implement the same refusal-plus-confidence gate and one focused "
            "rank-1 neighbor retry as a minimal production experiment.",
        )
    if lifecycle_complete:
        return (
            "B. PROMISING BUT NEEDS MORE WORK",
            "Keep production unchanged while investigating unnecessary retries or regressions.",
        )
    return (
        "C. NOT SUITABLE",
        "Keep production unchanged because confidence gating is not reliable enough.",
    )


def print_final_summary(records, threshold, loaded_models):
    supported = [record for record in records if record["case"]["supported"]]
    unsupported = [record for record in records if not record["case"]["supported"]]
    retries = [record for record in records if record["adaptive"]["retry_triggered"]]
    lifecycle = next(
        record for record in supported if record["case"]["question"] == LIFECYCLE_QUESTION
    )
    direct_lake = next(
        record
        for record in supported
        if record["case"]["topic"] == "Direct Lake overview"
    )
    final_classification, recommendation = classification(records)
    lifecycle_final = normalize(lifecycle["adaptive"]["final_answer"])
    lifecycle_complete = all(
        normalize(stage) in lifecycle_final for stage in LIFECYCLE_STAGES
    )

    print("\n" + "=" * 100)
    print("ADAPTIVE CONTEXT EXPANSION EVALUATION")
    print("=" * 100)
    print(f"\nThreshold:\nTop-1 similarity >= {threshold:.6f}")
    print("\nTopic | Top-1 | Retry | Final context | Result")
    print("-" * 100)
    for record in records:
        adaptive = record["adaptive"]
        print(
            f"{record['case']['topic']} | {adaptive['top1']:.6f} | "
            f"{'YES' if adaptive['retry_triggered'] else 'NO'} | "
            f"{adaptive['final_condition']} | {'PASS' if record['passed'] else 'FAIL'}"
        )
    print(
        f"\nSupported questions passed: {sum(r['passed'] for r in supported)} / "
        f"{len(supported)}"
    )
    print(
        f"Unsupported questions passed: {sum(r['passed'] for r in unsupported)} / "
        f"{len(unsupported)}"
    )
    print(f"Retries triggered: {len(retries)} / {len(records)}")
    print("\nLifecycle:")
    print(
        "Baseline complete: "
        f"{'NO' if lifecycle['adaptive']['baseline_insufficient'] else 'YES'}"
    )
    print(f"Adaptive final complete: {'YES' if lifecycle_complete else 'NO'}")
    print("\nDirect Lake:")
    print(
        "Retry triggered: "
        f"{'YES' if direct_lake['adaptive']['retry_triggered'] else 'NO'}"
    )
    print(f"Result: {'PASS' if direct_lake['passed'] else 'FAIL'}")
    print(
        "\nUnsupported retries: "
        f"{sum(r['adaptive']['retry_triggered'] for r in unsupported)}"
    )
    print(
        "Additional latency caused by retries: "
        f"{sum(r['adaptive']['retry_time'] for r in retries):.3f} s"
    )
    print("\nFINAL CLASSIFICATION:")
    print(final_classification)
    print(recommendation)
    print(f"\nLoaded models after cleanup: {loaded_models}")


def loaded_model_ids():
    manager = FoundryLocalManager.instance
    return (
        [model.id for model in manager.catalog.get_loaded_models()]
        if manager is not None
        else []
    )


def main():
    require_baseline_corpus()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    retrieval_cache = {}
    similarity_rows = collect_coverage_retrievals(retrieval_cache)
    print_similarity_table(similarity_rows)
    threshold_analysis = derive_threshold(similarity_rows)
    print_threshold_analysis(threshold_analysis)
    if not threshold_analysis["defensible"]:
        print(
            "\nSimilarity does not provide a sufficiently wide empirical safety "
            "margin. Adaptive generation was not run."
        )
        print(f"Loaded models after cleanup: {loaded_model_ids()}")
        return

    threshold = threshold_analysis["threshold"]
    for case in INITIAL_CASES:
        retrieval = retrieve_once(case["question"], retrieval_cache)
        print_fixed_retrieval(case["topic"], retrieval)

    session = None
    full_records = []
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

        initial_records = []
        for case in INITIAL_CASES:
            retrieval = retrieve_once(case["question"], retrieval_cache)
            adaptive_result = adaptive_answer(
                session, case["question"], retrieval, threshold
            )
            checks, passed = evaluate_case(case, adaptive_result)
            behavior_passed = (
                passed
                and adaptive_result["retry_triggered"] == case["expected_retry"]
            )
            checks["expected_retry_behavior"] = (
                adaptive_result["retry_triggered"] == case["expected_retry"]
            )
            print_adaptive_case(
                "INITIAL ADAPTIVE TEST",
                case,
                adaptive_result,
                checks,
                behavior_passed,
            )
            initial_records.append(
                {
                    "case": case,
                    "adaptive": adaptive_result,
                    "checks": checks,
                    "passed": behavior_passed,
                }
            )

        if not initial_cases_passed(initial_records):
            print(
                "\nInitial adaptive cases did not all meet expected behavior. "
                "The full coverage run was not started."
            )
            return

        initial_by_question = {
            record["case"]["question"]: record for record in initial_records
        }
        full_records = evaluate_full_coverage(
            session,
            retrieval_cache,
            threshold,
            initial_by_question,
        )
    finally:
        if session is not None:
            session.close()

    print_final_summary(full_records, threshold, loaded_model_ids())


if __name__ == "__main__":
    main()
