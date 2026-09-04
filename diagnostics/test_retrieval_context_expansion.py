import re
import sys
import time

from foundry_local_sdk import FoundryLocalManager

from fabric_rag.rag import RAGSession, build_context, build_messages
from fabric_rag.retrieval import get_top_chunks
from .retrieval_performance_baseline import (
    BASELINE_CHUNK_COUNT,
    require_baseline_corpus,
)


LIFECYCLE_QUESTION = (
    "What are the six stages of the end-to-end data lifecycle in Microsoft Fabric?"
)
REGRESSION_CASES = [
    ("OneLake", "What is OneLake?", True),
    ("Lakehouse", "What is a lakehouse in Microsoft Fabric?", True),
    ("Data Factory", "What is Data Factory in Microsoft Fabric?", True),
    ("Direct Lake", "How does Direct Lake work?", True),
    ("Unsupported", "Who is the CEO of Microsoft?", False),
]
LIFECYCLE_SOURCE = "data-lifecycle.md"
LIFECYCLE_STAGES = [
    "Get data",
    "Store data",
    "Prepare and transform",
    "Analyze and train",
    "Track and visualize",
    "External integration",
]
CITATION_PATTERN = re.compile(r"\[([0-9]+)\]")
REFUSAL_PATTERN = re.compile(
    r"(?:not enough information|does not (?:contain|include|provide)|"
    r"cannot (?:answer|determine)|unable to (?:answer|determine))",
    re.IGNORECASE,
)
MARKDOWN_URL_PATTERN = re.compile(r"\[[^\]]+\]\([^\)]+\)|https?://|www\.")


def lifecycle_stages_in(text):
    folded = text.casefold()
    return [stage for stage in LIFECYCLE_STAGES if stage.casefold() in folded]


def largest_overlap(left, right, limit=500):
    maximum = min(limit, len(left), len(right))
    for size in range(maximum, 0, -1):
        if left[-size:] == right[:size]:
            return size
    return 0


def merge_ordered_chunks(chunks):
    ordered = sorted(chunks, key=lambda chunk: chunk["chunk_number"])
    merged = ordered[0]["text"]
    removed_overlap = 0
    for chunk in ordered[1:]:
        overlap = largest_overlap(merged, chunk["text"])
        removed_overlap += overlap
        separator = "" if overlap else "\n\n"
        merged = merged + separator + chunk["text"][overlap:]
    return merged, removed_overlap


def duplicate_paragraphs(text):
    seen = set()
    duplicates = []
    for paragraph in re.split(r"\n\s*\n", text):
        normalized = " ".join(paragraph.split()).casefold()
        if len(normalized) < 40:
            continue
        if normalized in seen and normalized not in duplicates:
            duplicates.append(normalized)
        seen.add(normalized)
    return duplicates


def adjacent_candidates(primary, ranked_chunks, baseline):
    baseline_keys = {
        (chunk["source"], chunk["chunk_number"]) for chunk in baseline
    }
    candidates = []
    for chunk in ranked_chunks:
        same_source = chunk["source"] == primary["source"]
        adjacent = abs(chunk["chunk_number"] - primary["chunk_number"]) == 1
        key = (chunk["source"], chunk["chunk_number"])
        if same_source and adjacent and key not in baseline_keys:
            candidates.append(chunk)
    return sorted(candidates, key=lambda chunk: chunk["chunk_number"])


def expand_rank_one(baseline, neighbors):
    if not neighbors:
        return [dict(chunk) for chunk in baseline], 0

    primary = baseline[0]
    members = [primary, *neighbors]
    merged_text, removed_overlap = merge_ordered_chunks(members)
    member_numbers = sorted(chunk["chunk_number"] for chunk in members)
    expanded_primary = dict(primary)
    expanded_primary["text"] = merged_text
    expanded_primary["chunk_number"] = ", ".join(map(str, member_numbers))
    expanded_primary["expanded_chunk_numbers"] = member_numbers
    return [expanded_primary, *[dict(chunk) for chunk in baseline[1:]]], removed_overlap


def prepare_strategies(ranked_chunks):
    baseline = [dict(chunk) for chunk in ranked_chunks[:3]]
    candidates = adjacent_candidates(baseline[0], ranked_chunks, baseline)
    most_relevant = (
        [max(candidates, key=lambda chunk: chunk["similarity"])]
        if candidates
        else []
    )

    both, both_overlap = expand_rank_one(baseline, candidates)
    focused, focused_overlap = expand_rank_one(baseline, most_relevant)
    return {
        "A. Baseline Top-3 only": {
            "chunks": baseline,
            "neighbors": [],
            "overlap_removed": 0,
        },
        "B. Top-3 plus both immediate rank-1 neighbors": {
            "chunks": both,
            "neighbors": candidates,
            "overlap_removed": both_overlap,
        },
        "C. Top-3 plus most relevant rank-1 neighbor": {
            "chunks": focused,
            "neighbors": most_relevant,
            "overlap_removed": focused_overlap,
        },
    }


def strategy_metrics(strategy):
    context = build_context(strategy["chunks"])
    return {
        "context": context,
        "characters": len(context),
        "stages": lifecycle_stages_in(context),
        "complete": all(
            stage.casefold() in context.casefold() for stage in LIFECYCLE_STAGES
        ),
        "duplicates": duplicate_paragraphs(context),
    }


def choose_best_strategy(strategies):
    focused_name = "C. Top-3 plus most relevant rank-1 neighbor"
    focused_metrics = strategy_metrics(strategies[focused_name])
    if focused_metrics["complete"]:
        return focused_name

    both_name = "B. Top-3 plus both immediate rank-1 neighbors"
    both_metrics = strategy_metrics(strategies[both_name])
    return both_name if both_metrics["complete"] else None


def generate_answer(session, question, chunks):
    context = build_context(chunks)
    start = time.perf_counter()
    completion = session._chat_client.complete_chat(build_messages(question, context))
    elapsed = time.perf_counter() - start
    return completion.choices[0].message.content, elapsed


def citation_checks(answer, chunk_count=3):
    citations = CITATION_PATTERN.findall(answer or "")
    valid = {str(rank) for rank in range(1, chunk_count + 1)}
    return citations, set(citations).issubset(valid)


def print_ranked_chunks(chunks):
    for rank, chunk in enumerate(chunks, start=1):
        print(
            f"Rank {rank}: {chunk['source']} chunk {chunk['chunk_number']} | "
            f"similarity {chunk['similarity']:.6f}"
        )


def print_lifecycle_diagnostic(ranked_chunks):
    print("\n" + "=" * 80)
    print("LIFECYCLE CHUNK DIAGNOSTIC")
    print("=" * 80)
    lifecycle = [
        chunk for chunk in ranked_chunks if chunk["source"] == LIFECYCLE_SOURCE
    ]
    for chunk in sorted(lifecycle, key=lambda item: item["chunk_number"]):
        stages = lifecycle_stages_in(chunk["text"])
        ranking = ranked_chunks.index(chunk) + 1
        print(
            f"{chunk['source']} | chunk {chunk['chunk_number']} | "
            f"overall rank {ranking} | similarity {chunk['similarity']:.6f} | "
            f"stages: {', '.join(stages) or 'none'}"
        )


def print_strategy_report(strategies):
    print("\n" + "=" * 80)
    print("CONTEXT STRATEGIES")
    print("=" * 80)
    baseline_size = strategy_metrics(strategies["A. Baseline Top-3 only"])[
        "characters"
    ]
    for name, strategy in strategies.items():
        metrics = strategy_metrics(strategy)
        neighbors = ", ".join(
            f"{chunk['source']} chunk {chunk['chunk_number']} "
            f"({chunk['similarity']:.6f})"
            for chunk in strategy["neighbors"]
        ) or "none"
        print(f"\n{name}")
        print(f"Adjacent chunks included under source label [1]: {neighbors}")
        print(f"Context characters: {metrics['characters']}")
        print(f"Increase over baseline: {metrics['characters'] - baseline_size}")
        print(f"Overlap characters removed while merging: {strategy['overlap_removed']}")
        print(f"All six stages present: {'YES' if metrics['complete'] else 'NO'}")
        print(f"Stages present: {', '.join(metrics['stages']) or 'none'}")
        print(
            "Duplicate passages after merge: "
            f"{'YES' if metrics['duplicates'] else 'NO'}"
        )
        print("Source labels remain [1]-[3]: YES")


def print_comparison(topic, question, baseline_chunks, expanded_chunks, baseline_answer,
                     expanded_answer, baseline_time, expanded_time, supported):
    baseline_citations, baseline_valid = citation_checks(baseline_answer)
    expanded_citations, expanded_valid = citation_checks(expanded_answer)
    print("\n" + "=" * 80)
    print(f"REGRESSION COMPARISON: {topic}")
    print("=" * 80)
    print(f"Question: {question}")
    print("\nSemantic Top-3 (unchanged):")
    print_ranked_chunks(baseline_chunks)
    print(
        "Expanded rank-1 chunks: "
        f"{expanded_chunks[0].get('expanded_chunk_numbers', [baseline_chunks[0]['chunk_number']])}"
    )
    print(f"Baseline answer ({baseline_time:.3f} seconds):\n{baseline_answer}")
    print(f"\nExpanded answer ({expanded_time:.3f} seconds):\n{expanded_answer}")
    print(
        "Baseline citations: "
        f"{baseline_citations or 'none'} | valid: {'YES' if baseline_valid else 'NO'}"
    )
    print(
        "Expanded citations: "
        f"{expanded_citations or 'none'} | valid: {'YES' if expanded_valid else 'NO'}"
    )
    print(
        "Expanded answer contains Markdown URL: "
        f"{'YES' if MARKDOWN_URL_PATTERN.search(expanded_answer or '') else 'NO'}"
    )
    if supported:
        print(
            "Expanded answer refused: "
            f"{'YES' if REFUSAL_PATTERN.search(expanded_answer or '') else 'NO'}"
        )
    else:
        print(
            "Expanded answer refused: "
            f"{'YES' if REFUSAL_PATTERN.search(expanded_answer or '') else 'NO'}"
        )
        print(
            "Expanded refusal has no citations: "
            f"{'YES' if not expanded_citations else 'NO'}"
        )


def main():
    require_baseline_corpus()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    lifecycle_ranked = get_top_chunks(
        LIFECYCLE_QUESTION,
        top_k=BASELINE_CHUNK_COUNT,
    )
    lifecycle_strategies = prepare_strategies(lifecycle_ranked)
    best_name = choose_best_strategy(lifecycle_strategies)

    print_lifecycle_diagnostic(lifecycle_ranked)

    baseline = lifecycle_strategies["A. Baseline Top-3 only"]
    print("\n" + "=" * 80)
    print("BASELINE")
    print("=" * 80)
    print(f"Question: {LIFECYCLE_QUESTION}")
    print_ranked_chunks(baseline["chunks"])
    baseline_metrics = strategy_metrics(baseline)
    print(f"All six stages available: {'YES' if baseline_metrics['complete'] else 'NO'}")
    print("\nBASELINE CONTEXT TEXT")
    print(baseline_metrics["context"])

    print_strategy_report(lifecycle_strategies)
    print(f"\nSelected experimental strategy: {best_name or 'none'}")
    if best_name is None:
        print("No expanded strategy supplied all six lifecycle stages; generation skipped.")
        return

    session = None
    try:
        session = RAGSession()
        print("\n" + "=" * 80)
        print("LIFECYCLE ANSWER BEFORE AND AFTER")
        print("=" * 80)
        baseline_answer, baseline_time = generate_answer(
            session, LIFECYCLE_QUESTION, baseline["chunks"]
        )
        expanded = lifecycle_strategies[best_name]
        expanded_answer, expanded_time = generate_answer(
            session, LIFECYCLE_QUESTION, expanded["chunks"]
        )
        print(f"Baseline answer ({baseline_time:.3f} seconds):\n{baseline_answer}")
        print(f"\nExpanded answer ({expanded_time:.3f} seconds):\n{expanded_answer}")
        expanded_citations, expanded_valid = citation_checks(expanded_answer)
        print(f"\nExpanded citations: {expanded_citations or 'none'}")
        print(f"Expanded citation labels valid: {'YES' if expanded_valid else 'NO'}")
        print(
            "Expanded answer mentions all six stages: "
            f"{'YES' if len(lifecycle_stages_in(expanded_answer or '')) == 6 else 'NO'}"
        )

        for topic, question, supported in REGRESSION_CASES:
            ranked = get_top_chunks(question, top_k=BASELINE_CHUNK_COUNT)
            strategies = prepare_strategies(ranked)
            baseline_case = strategies["A. Baseline Top-3 only"]["chunks"]
            expanded_case = strategies[best_name]["chunks"]
            baseline_case_answer, baseline_case_time = generate_answer(
                session, question, baseline_case
            )
            expanded_case_answer, expanded_case_time = generate_answer(
                session, question, expanded_case
            )
            print_comparison(
                topic,
                question,
                baseline_case,
                expanded_case,
                baseline_case_answer,
                expanded_case_answer,
                baseline_case_time,
                expanded_case_time,
                supported,
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
    print("\n" + "=" * 80)
    print("DIAGNOSTIC CLEANUP")
    print("=" * 80)
    print(f"Loaded models after cleanup: {loaded_models}")


if __name__ == "__main__":
    main()
