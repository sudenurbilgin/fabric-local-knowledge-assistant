import hashlib
import re
import sys
import time
from pathlib import Path
from statistics import mean

from foundry_local_sdk import FoundryLocalManager

from fabric_rag.rag import RAGSession, build_context, build_messages
from .test_adaptive_context_expansion import (
    INSUFFICIENT_CONTEXT_PATTERN,
    LIFECYCLE_QUESTION,
    LIFECYCLE_STAGES,
    generate,
    retrieve_once,
)
from .test_adaptive_holdout import UNSUPPORTED_HOLDOUT


ADAPTIVE_THRESHOLD = 0.528094
EXPECTED_MODEL = "Phi-4-mini-instruct-generic-cpu:5"
EXPECTED_PROVIDER = "CPUExecutionProvider"
LIFECYCLE_REPETITIONS = 3
CITATION_PATTERN = re.compile(r"\[([0-9]+)\]")
MARKDOWN_URL_PATTERN = re.compile(r"\[[^\]]+\]\([^\)]+\)|https?://|www\.")
SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?:\r?\n)+|(?<=[.!?])\s+")
WORD_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9'-]*")
PROTECTED_FILES = [
    "src/fabric_rag/rag.py",
    "src/fabric_rag/retrieval.py",
    "streamlit_app.py",
    "rag.db",
    "src/fabric_rag/documents.py",
    "src/fabric_rag/knowledge_base.py",
]

NORMAL_SUPPORTED = [
    {
        "topic": "OneLake",
        "question": "What is OneLake?",
        "answer_groups": [["OneLake"], ["data lake", "unified"]],
    },
    {
        "topic": "Lakehouse",
        "question": "What is a lakehouse in Microsoft Fabric?",
        "answer_groups": [["lakehouse"], ["data lake"], ["warehouse"]],
    },
    {
        "topic": "Data Factory",
        "question": "What is Data Factory in Microsoft Fabric?",
        "answer_groups": [["Data Factory"], ["data integration", "integrate data"]],
    },
    {
        "topic": "Direct Lake",
        "question": "How does Direct Lake work?",
        "answer_groups": [["Direct Lake"], ["OneLake", "Delta"], ["VertiPaq", "memory"]],
    },
]

DISTANT_UNSUPPORTED = [
    {
        "topic": "Microsoft leadership",
        "question": "Who is the CEO of Microsoft?",
        "forbidden": ["Satya Nadella"],
    },
    {
        "topic": "Current weather",
        "question": "What is the current weather in Istanbul?",
        "forbidden": ["sunny", "cloudy", "raining", "degrees", "°C", "°F"],
    },
    {
        "topic": "Unrelated geography",
        "question": "What is the capital of Japan?",
        "forbidden": ["Tokyo"],
    },
]


def normalize(text):
    return " ".join(text.casefold().split())


def file_hashes():
    return {
        filename: hashlib.sha256(Path(filename).read_bytes()).hexdigest()
        for filename in PROTECTED_FILES
    }


def citation_analysis(answer, chunk_count=3):
    citations = CITATION_PATTERN.findall(answer)
    valid_labels = {str(rank) for rank in range(1, chunk_count + 1)}
    valid_citations = [label for label in citations if label in valid_labels]
    return {
        "citations": citations,
        "valid_citations": valid_citations,
        "valid_citations_present": bool(valid_citations),
        "all_citations_valid": bool(citations) and set(citations).issubset(valid_labels),
    }


def is_substantive_non_refusal(sentence):
    if INSUFFICIENT_CONTEXT_PATTERN.search(sentence):
        return False
    cleaned = CITATION_PATTERN.sub("", sentence)
    cleaned = re.sub(r"[*_#>`-]", " ", cleaned)
    cleaned = re.sub(
        r"^(?:according to|based on) (?:the )?(?:supplied|provided) context[:,]?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return len(WORD_PATTERN.findall(cleaned)) >= 8


def mixed_partial_refusal(answer):
    sentences = [
        sentence.strip()
        for sentence in SENTENCE_BOUNDARY_PATTERN.split(answer)
        if sentence.strip()
    ]
    refusal_present = any(
        INSUFFICIENT_CONTEXT_PATTERN.search(sentence) is not None
        for sentence in sentences
    )
    substantive_present = any(
        is_substantive_non_refusal(sentence) for sentence in sentences
    )
    return refusal_present and substantive_present


def analyze_baseline(answer, retrieval):
    citation = citation_analysis(answer, len(retrieval["baseline_chunks"]))
    insufficient = INSUFFICIENT_CONTEXT_PATTERN.search(answer) is not None
    high_similarity = (
        retrieval["baseline_chunks"][0]["similarity"] >= ADAPTIVE_THRESHOLD
    )
    neighbor_available = bool(retrieval["neighbor"])
    current_retry = insufficient and high_similarity and neighbor_available
    refined_retry = current_retry and citation["valid_citations_present"]
    return {
        **citation,
        "insufficient": insufficient,
        "high_similarity": high_similarity,
        "neighbor_available": neighbor_available,
        "mixed_partial_refusal": mixed_partial_refusal(answer),
        "current_retry": current_retry,
        "refined_retry": refined_retry,
    }


def generate_with_refined_gate(session, question, retrieval):
    baseline_messages = build_messages(
        question, build_context(retrieval["baseline_chunks"])
    )
    baseline_answer, baseline_time = generate(session, baseline_messages)
    shape = analyze_baseline(baseline_answer, retrieval)

    final_answer = baseline_answer
    final_chunks = retrieval["baseline_chunks"]
    retry_time = 0.0
    if shape["refined_retry"]:
        expanded_messages = build_messages(
            question, build_context(retrieval["expanded_chunks"])
        )
        final_answer, retry_time = generate(session, expanded_messages)
        final_chunks = retrieval["expanded_chunks"]

    return {
        "question": question,
        "retrieval": retrieval,
        "top1": retrieval["baseline_chunks"][0]["similarity"],
        "baseline_answer": baseline_answer,
        "baseline_time": baseline_time,
        "shape": shape,
        "final_answer": final_answer,
        "final_chunks": final_chunks,
        "retry_time": retry_time,
        "total_time": baseline_time + retry_time,
        "final_context": "EXPANDED" if shape["refined_retry"] else "BASELINE",
    }


def lifecycle_stage_presence(answer):
    normalized = normalize(answer)
    return [stage for stage in LIFECYCLE_STAGES if normalize(stage) in normalized]


def evaluate_lifecycle(result):
    baseline_stages = lifecycle_stage_presence(result["baseline_answer"])
    final_stages = lifecycle_stage_presence(result["final_answer"])
    final_citations = citation_analysis(
        result["final_answer"], len(result["final_chunks"])
    )
    final_insufficient = (
        INSUFFICIENT_CONTEXT_PATTERN.search(result["final_answer"]) is not None
    )
    useful_retry = (
        result["shape"]["refined_retry"]
        and len(baseline_stages) < len(LIFECYCLE_STAGES)
        and len(final_stages) == len(LIFECYCLE_STAGES)
    )
    checks = {
        "baseline_incomplete": len(baseline_stages) < len(LIFECYCLE_STAGES),
        "baseline_insufficient_signal": result["shape"]["insufficient"],
        "baseline_valid_citation_present": result["shape"][
            "valid_citations_present"
        ],
        "refined_retry_triggered": result["shape"]["refined_retry"],
        "all_six_stages_present": len(final_stages) == len(LIFECYCLE_STAGES),
        "final_citations_valid": final_citations["all_citations_valid"],
        "no_final_insufficient_signal": not final_insufficient,
        "no_markdown_urls": MARKDOWN_URL_PATTERN.search(result["final_answer"]) is None,
        "useful_retry": useful_retry,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "baseline_stages": baseline_stages,
        "final_stages": final_stages,
        "final_citations": final_citations,
        "useful_retry": useful_retry,
    }


def evaluate_supported(case, result):
    normalized_answer = normalize(result["final_answer"])
    citations = citation_analysis(result["final_answer"], len(result["final_chunks"]))
    checks = {
        "substantive_answer": len(WORD_PATTERN.findall(result["final_answer"])) >= 20,
        "expected_concepts_present": all(
            any(normalize(term) in normalized_answer for term in alternatives)
            for alternatives in case["answer_groups"]
        ),
        "no_insufficient_signal": (
            INSUFFICIENT_CONTEXT_PATTERN.search(result["final_answer"]) is None
        ),
        "citations_present_and_valid": citations["all_citations_valid"],
        "no_markdown_urls": MARKDOWN_URL_PATTERN.search(result["final_answer"]) is None,
        "no_retry": not result["shape"]["refined_retry"],
    }
    return checks, all(checks.values())


def evaluate_near_unsupported(case, result):
    answer = result["final_answer"]
    citations = CITATION_PATTERN.findall(answer)
    unsafe_assertions = [
        pattern
        for pattern in case["unsafe_patterns"]
        if re.search(pattern, answer, re.IGNORECASE | re.DOTALL)
    ]
    checks = {
        "clear_refusal": INSUFFICIENT_CONTEXT_PATTERN.search(answer) is not None,
        "no_citations": not citations,
        "no_unsupported_fact": not unsafe_assertions,
        "no_markdown_urls": MARKDOWN_URL_PATTERN.search(answer) is None,
    }
    return checks, all(checks.values()), unsafe_assertions


def evaluate_distant_unsupported(case, result):
    answer = result["final_answer"]
    folded = answer.casefold()
    checks = {
        "clear_refusal": INSUFFICIENT_CONTEXT_PATTERN.search(answer) is not None,
        "requested_fact_not_given": all(
            forbidden.casefold() not in folded for forbidden in case["forbidden"]
        ),
        "no_citations": not CITATION_PATTERN.findall(answer),
        "no_markdown_urls": MARKDOWN_URL_PATTERN.search(answer) is None,
        "no_retry": not result["shape"]["refined_retry"],
    }
    return checks, all(checks.values())


def print_retrieval(result):
    print("Top-3 retrieval:")
    for rank, chunk in enumerate(result["retrieval"]["baseline_chunks"], start=1):
        print(
            f"  {rank}. {chunk['source']} chunk {chunk['chunk_number']} | "
            f"{chunk['similarity']:.6f}"
        )
    neighbor = result["retrieval"]["neighbor"]
    if neighbor:
        chunk = neighbor[0]
        print(
            f"Focused neighbor: {chunk['source']} chunk {chunk['chunk_number']} | "
            f"{chunk['similarity']:.6f}"
        )
    else:
        print("Focused neighbor: none")


def print_shape(result):
    shape = result["shape"]
    citation_text = ", ".join(
        f"[{label}]" for label in shape["citations"]
    ) or "None"
    print(f"Insufficient signal: {'YES' if shape['insufficient'] else 'NO'}")
    print(
        "Valid citations present: "
        f"{'YES' if shape['valid_citations_present'] else 'NO'} "
        f"({citation_text})"
    )
    print(
        "Mixed partial/refusal: "
        f"{'YES' if shape['mixed_partial_refusal'] else 'NO'}"
    )
    print(f"Current gate retry: {'YES' if shape['current_retry'] else 'NO'}")
    print(f"Refined gate retry: {'YES' if shape['refined_retry'] else 'NO'}")
    print(f"Final context: {result['final_context']}")
    print(f"Baseline generation time: {result['baseline_time']:.3f} s")
    print(f"Retry generation time: {result['retry_time']:.3f} s")


def print_case_header(category, topic, question, result):
    print("\n" + "=" * 100)
    print(f"{category}: {topic}")
    print("=" * 100)
    print(f"Question: {question}")
    print_retrieval(result)
    print_shape(result)
    print("\nBASELINE ANSWER")
    print(result["baseline_answer"])
    if result["shape"]["refined_retry"]:
        print("\nEXPANDED ANSWER")
        print(result["final_answer"])


def print_checks(checks, passed):
    print("\nACCEPTANCE CHECKS")
    for name, value in checks.items():
        print(f"- {name}: {'PASS' if value else 'FAIL'}")
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")


def run_lifecycle(session, retrieval):
    records = []
    for run_number in range(1, LIFECYCLE_REPETITIONS + 1):
        result = generate_with_refined_gate(session, LIFECYCLE_QUESTION, retrieval)
        evaluation = evaluate_lifecycle(result)
        print_case_header(
            f"LIFECYCLE RUN {run_number}",
            "Lifecycle",
            LIFECYCLE_QUESTION,
            result,
        )
        print(
            "Baseline stages: "
            + (", ".join(evaluation["baseline_stages"]) or "None")
        )
        print(
            "Final stages: "
            + (", ".join(evaluation["final_stages"]) or "None")
        )
        print_checks(evaluation["checks"], evaluation["passed"])
        records.append(
            {
                "run": run_number,
                "topic": "Lifecycle",
                "result": result,
                **evaluation,
            }
        )
    return records


def run_near_unsupported(session, retrieval_cache):
    records = []
    for case in UNSUPPORTED_HOLDOUT:
        retrieval = retrieve_once(case["question"], retrieval_cache)
        result = generate_with_refined_gate(session, case["question"], retrieval)
        checks, passed, unsafe_assertions = evaluate_near_unsupported(case, result)
        print_case_header(
            "NEAR-DOMAIN UNSUPPORTED", case["topic"], case["question"], result
        )
        print(f"Corpus reason: {case['reason']}")
        if unsafe_assertions:
            print(f"Detected unsupported patterns: {unsafe_assertions}")
        print_checks(checks, passed)
        records.append(
            {
                "case": case,
                "topic": case["topic"],
                "result": result,
                "checks": checks,
                "passed": passed,
                "unsafe_assertions": unsafe_assertions,
                "useful_retry": False,
            }
        )
    return records


def run_normal_supported(session, retrieval_cache):
    records = []
    for case in NORMAL_SUPPORTED:
        retrieval = retrieve_once(case["question"], retrieval_cache)
        result = generate_with_refined_gate(session, case["question"], retrieval)
        checks, passed = evaluate_supported(case, result)
        print_case_header(
            "NORMAL SUPPORTED", case["topic"], case["question"], result
        )
        print_checks(checks, passed)
        records.append(
            {
                "case": case,
                "topic": case["topic"],
                "result": result,
                "checks": checks,
                "passed": passed,
                "useful_retry": False,
            }
        )
    return records


def run_distant_unsupported(session, retrieval_cache):
    records = []
    for case in DISTANT_UNSUPPORTED:
        retrieval = retrieve_once(case["question"], retrieval_cache)
        result = generate_with_refined_gate(session, case["question"], retrieval)
        checks, passed = evaluate_distant_unsupported(case, result)
        print_case_header(
            "DISTANT UNSUPPORTED", case["topic"], case["question"], result
        )
        print_checks(checks, passed)
        records.append(
            {
                "case": case,
                "topic": case["topic"],
                "result": result,
                "checks": checks,
                "passed": passed,
                "useful_retry": False,
            }
        )
    return records


def current_retry_count(records):
    return sum(record["result"]["shape"]["current_retry"] for record in records)


def refined_retry_count(records):
    return sum(record["result"]["shape"]["refined_retry"] for record in records)


def estimated_avoided_latency(records):
    return sum(
        record["result"]["baseline_time"]
        for record in records
        if record["result"]["shape"]["current_retry"]
        and not record["result"]["shape"]["refined_retry"]
    )


def classify(lifecycle, near, normal, distant):
    lifecycle_useful = sum(record["useful_retry"] for record in lifecycle)
    near_current = current_retry_count(near)
    near_refined = refined_retry_count(near)
    unsafe_results = sum(not record["passed"] for record in near + distant)
    ordinary_clean = all(
        record["passed"] and not record["result"]["shape"]["refined_retry"]
        for record in normal
    )
    distant_clean = all(
        record["passed"] and not record["result"]["shape"]["refined_retry"]
        for record in distant
    )
    if (
        lifecycle_useful == LIFECYCLE_REPETITIONS
        and near_refined == 0
        and unsafe_results == 0
        and ordinary_clean
        and distant_clean
    ):
        return (
            "A. REFINED GATE VALIDATED",
            "The refined gate is a candidate for minimal production integration.",
        )
    if (
        lifecycle_useful > 0
        and near_refined < near_current
        and unsafe_results == 0
    ):
        return (
            "B. REFINED GATE PROMISING BUT INCOMPLETE",
            "Keep production unchanged and investigate one further gate refinement.",
        )
    return (
        "C. REFINED GATE NOT RELIABLE",
        "Keep production unchanged because the refined gate is not dependable.",
    )


def loaded_models():
    manager = FoundryLocalManager.instance
    return (
        [model.id for model in manager.catalog.get_loaded_models()]
        if manager is not None
        else []
    )


def print_footer(lifecycle, near, normal, distant, hashes_match, model_ids):
    all_records = lifecycle + near + normal + distant
    lifecycle_useful = sum(record["useful_retry"] for record in lifecycle)
    near_current = current_retry_count(near)
    near_refined = refined_retry_count(near)
    distant_current = current_retry_count(distant)
    distant_refined = refined_retry_count(distant)
    old_useful = sum(
        record["useful_retry"]
        and record["result"]["shape"]["current_retry"]
        for record in lifecycle
    )
    refined_useful = sum(record["useful_retry"] for record in all_records)
    old_unnecessary = current_retry_count(all_records) - old_useful
    refined_unnecessary = refined_retry_count(all_records) - refined_useful
    unsafe_retries = sum(
        record["result"]["shape"]["refined_retry"] and not record["passed"]
        for record in near + distant
    )
    avoided = estimated_avoided_latency(all_records)
    classification, recommendation = classify(lifecycle, near, normal, distant)

    print("\n" + "=" * 100)
    print("REFINED ADAPTIVE GATE EVALUATION")
    print("=" * 100)
    print(f"\nFrozen threshold: {ADAPTIVE_THRESHOLD:.6f}")

    print("\nLIFECYCLE REPEATABILITY")
    print("Run | Citations | Mixed | Retry | Final result")
    for record in lifecycle:
        shape = record["result"]["shape"]
        print(
            f"Run {record['run']} | "
            f"{'YES' if shape['valid_citations_present'] else 'NO'} | "
            f"{'YES' if shape['mixed_partial_refusal'] else 'NO'} | "
            f"{'YES' if shape['refined_retry'] else 'NO'} | "
            f"{'PASS' if record['passed'] else 'FAIL'}"
        )
    print(f"Useful lifecycle retries: {lifecycle_useful} / 3")

    print("\nNEAR-DOMAIN UNSUPPORTED")
    print("Topic | Current retry | Refined retry | Result")
    for record in near:
        shape = record["result"]["shape"]
        print(
            f"{record['topic']} | "
            f"{'YES' if shape['current_retry'] else 'NO'} | "
            f"{'YES' if shape['refined_retry'] else 'NO'} | "
            f"{'PASS' if record['passed'] else 'FAIL'}"
        )
    print(f"Refined unsupported retries: {near_refined} / 8")

    print("\nNORMAL SUPPORTED")
    print("Topic | Retry | Result")
    for record in normal:
        print(
            f"{record['topic']} | "
            f"{'YES' if record['result']['shape']['refined_retry'] else 'NO'} | "
            f"{'PASS' if record['passed'] else 'FAIL'}"
        )

    print("\nDISTANT UNSUPPORTED")
    print("Topic | Retry | Result")
    for record in distant:
        print(
            f"{record['topic']} | "
            f"{'YES' if record['result']['shape']['refined_retry'] else 'NO'} | "
            f"{'PASS' if record['passed'] else 'FAIL'}"
        )

    print("\nCURRENT GATE")
    print(f"Lifecycle useful retries: {old_useful} / 3")
    print(f"Near-domain unsupported retries: {near_current} / 8")
    print(f"Other unsupported retries: {distant_current} / 3")
    print(f"Unnecessary retries: {old_unnecessary}")
    print("Unsafe retries: not re-executed for current-only decisions")

    print("\nREFINED GATE")
    print(f"Lifecycle useful retries: {lifecycle_useful} / 3")
    print(f"Near-domain unsupported retries: {near_refined} / 8")
    print(f"Other unsupported retries: {distant_refined} / 3")
    print(f"Unnecessary retries: {refined_unnecessary}")
    print(f"Unsafe retries: {unsafe_retries}")

    prevented = old_unnecessary - refined_unnecessary
    print(f"\nUnnecessary retries eliminated: {prevented}")
    print(f"Estimated retry latency avoided: {avoided:.3f} seconds")
    print(
        "Latency estimate method: sum of same-run baseline latencies for "
        "current-gate retries blocked by the refined gate."
    )
    print(f"Unsafe retries: {unsafe_retries}")

    print("\nFINAL CLASSIFICATION")
    print(classification)
    print(recommendation)
    print(f"\nProtected hashes unchanged: {'YES' if hashes_match else 'NO'}")
    print(f"Loaded models after cleanup: {model_ids}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    hashes_before = file_hashes()
    print("PROTECTED SHA-256 HASHES BEFORE")
    for filename, digest in hashes_before.items():
        print(f"{filename}: {digest}")

    retrieval_cache = {}
    lifecycle_retrieval = retrieve_once(LIFECYCLE_QUESTION, retrieval_cache)
    for case in NORMAL_SUPPORTED + DISTANT_UNSUPPORTED:
        retrieve_once(case["question"], retrieval_cache)
    for case in UNSUPPORTED_HOLDOUT:
        retrieve_once(case["question"], retrieval_cache)

    session = None
    lifecycle = []
    near = []
    normal = []
    distant = []
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

        lifecycle = run_lifecycle(session, lifecycle_retrieval)
        near = run_near_unsupported(session, retrieval_cache)
        normal = run_normal_supported(session, retrieval_cache)
        distant = run_distant_unsupported(session, retrieval_cache)
    finally:
        if session is not None:
            session.close()

    hashes_after = file_hashes()
    hashes_match = hashes_before == hashes_after
    print("\nPROTECTED SHA-256 HASHES AFTER")
    for filename, digest in hashes_after.items():
        status = "UNCHANGED" if digest == hashes_before[filename] else "CHANGED"
        print(f"{filename}: {digest} | {status}")

    print_footer(
        lifecycle,
        near,
        normal,
        distant,
        hashes_match,
        loaded_models(),
    )


if __name__ == "__main__":
    main()
