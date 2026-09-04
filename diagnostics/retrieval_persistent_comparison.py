import sys
import time
from collections import defaultdict
from contextlib import ExitStack
from statistics import mean
from unittest.mock import patch

from foundry_local_sdk import FoundryLocalManager

from fabric_rag import retrieval
from fabric_rag.config import (
    EXPECTED_EMBEDDING_DIMENSIONS,
    RETRIEVAL_TOP_K,
)
from diagnostics.retrieval_performance_baseline import (
    BASELINE_CHUNK_COUNT,
    QUERIES,
    TimedEmbeddingModel,
    TimedJsonModule,
    measure,
    measure_retrieval,
    require_baseline_corpus,
    validate_run,
)


STATELESS_REPETITIONS = 2
PERSISTENT_REPETITIONS = 3
SIMILARITY_TOLERANCE = 1e-9
EXPECTED_RESULTS = {
    "What is OneLake?": (
        ("onelake-overview.md", 1, 0.799319316627),
        ("microsoft-fabric-overview.md", 11, 0.749994069988),
        ("onelake-overview.md", 4, 0.710338871704),
    ),
    "What is Data Factory in Microsoft Fabric?": (
        ("microsoft-fabric-overview.md", 9, 0.858649045661),
        ("microsoft-fabric-overview.md", 1, 0.798565946801),
        ("microsoft-fabric-overview.md", 7, 0.786757658591),
    ),
    "How does Direct Lake work?": (
        ("direct-lake-overview.md", 1, 0.774049407836),
        ("direct-lake-overview.md", 16, 0.670768691103),
        ("direct-lake-overview.md", 3, 0.657171064737),
    ),
    "Who is the CEO of Microsoft?": (
        ("microsoft-fabric-overview.md", 9, 0.353604588208),
        ("microsoft-fabric-overview.md", 5, 0.321000120321),
        ("microsoft-fabric-overview.md", 1, 0.315260819837),
    ),
}


def validate_results(query, results, path_name):
    expected = EXPECTED_RESULTS[query]
    if len(results) != RETRIEVAL_TOP_K:
        raise RuntimeError(
            f"{path_name} returned {len(results)} results for {query!r}."
        )

    for rank, (actual, expected_result) in enumerate(
        zip(results, expected),
        start=1,
    ):
        expected_source, expected_chunk, expected_similarity = expected_result
        if (
            actual["source"] != expected_source
            or actual["chunk_number"] != expected_chunk
        ):
            raise RuntimeError(
                f"{path_name} rank {rank} changed for {query!r}: "
                f"{actual['source']} chunk {actual['chunk_number']}."
            )
        difference = abs(actual["similarity"] - expected_similarity)
        if difference > SIMILARITY_TOLERANCE:
            raise RuntimeError(
                f"{path_name} rank {rank} similarity changed by {difference:.12g} "
                f"for {query!r}."
            )


def run_stateless_comparison():
    runs_by_query = {query: [] for query in QUERIES}
    print("STATELESS get_top_chunks()")
    for repetition in range(1, STATELESS_REPETITIONS + 1):
        for query in QUERIES:
            run = measure_retrieval(query)
            validate_run(run)
            validate_results(query, run["results"], "Stateless retrieval")
            runs_by_query[query].append(run)
            print(
                f"Round {repetition} | {query} | "
                f"{run['timings']['total_retrieval']:.6f} seconds"
            )
    return runs_by_query


def run_persistent_comparison():
    timings = defaultdict(float)
    counters = defaultdict(int)
    dimensions = []
    runs_by_query = {query: [] for query in QUERIES}

    original_load_stored_chunks = retrieval.load_stored_chunks
    original_select_embedding_model = retrieval.select_embedding_model
    original_cosine_similarity = retrieval.cosine_similarity

    def timed_load_stored_chunks():
        counters["database_load_calls"] += 1
        with measure(timings, "stored_chunks_load"):
            return original_load_stored_chunks()

    def timed_select_embedding_model(manager):
        with measure(timings, "model_lookup_and_variant_selection"):
            model = original_select_embedding_model(manager)
        return TimedEmbeddingModel(model, timings, counters, dimensions)

    def timed_cosine_similarity(*args, **kwargs):
        counters["cosine_similarity_calls"] += 1
        with measure(timings, "cosine_similarity"):
            return original_cosine_similarity(*args, **kwargs)

    retriever = None
    cleanup_time = None
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                retrieval,
                "json",
                TimedJsonModule(retrieval.json, timings, counters),
            )
        )
        stack.enter_context(
            patch.object(
                retrieval,
                "load_stored_chunks",
                timed_load_stored_chunks,
            )
        )
        stack.enter_context(
            patch.object(
                retrieval,
                "select_embedding_model",
                timed_select_embedding_model,
            )
        )
        stack.enter_context(
            patch.object(
                retrieval,
                "cosine_similarity",
                timed_cosine_similarity,
            )
        )

        initialization_started = time.perf_counter()
        retriever = retrieval.PersistentRetriever()
        initialization_time = time.perf_counter() - initialization_started

        try:
            print("\nPERSISTENT PersistentRetriever")
            print(f"Initialization: {initialization_time:.6f} seconds")
            for repetition in range(1, PERSISTENT_REPETITIONS + 1):
                for query in QUERIES:
                    embedding_before = timings["query_embedding_generation"]
                    cosine_before = timings["cosine_similarity"]
                    started = time.perf_counter()
                    results = retriever.retrieve(query, top_k=RETRIEVAL_TOP_K)
                    elapsed = time.perf_counter() - started
                    query_embedding_time = (
                        timings["query_embedding_generation"] - embedding_before
                    )
                    cosine_time = timings["cosine_similarity"] - cosine_before
                    validate_results(query, results, "Persistent retrieval")
                    run = {
                        "results": results,
                        "total_retrieval": elapsed,
                        "query_embedding": query_embedding_time,
                        "cosine_similarity": cosine_time,
                    }
                    runs_by_query[query].append(run)
                    print(
                        f"Round {repetition} | {query} | total={elapsed:.6f}s | "
                        f"query embedding={query_embedding_time:.6f}s | "
                        f"cosine={cosine_time:.6f}s"
                    )
        finally:
            cleanup_started = time.perf_counter()
            retriever.close()
            retriever.close()
            cleanup_time = time.perf_counter() - cleanup_started

    expected_queries = len(QUERIES) * PERSISTENT_REPETITIONS
    expected_counts = {
        "database_load_calls": 1,
        "json_deserialization_calls": BASELINE_CHUNK_COUNT,
        "model_download_calls": 1,
        "model_load_calls": 1,
        "query_embedding_calls": expected_queries,
        "cosine_similarity_calls": BASELINE_CHUNK_COUNT * expected_queries,
        "model_unload_calls": 1,
    }
    for counter, expected in expected_counts.items():
        actual = counters.get(counter, 0)
        if actual != expected:
            raise RuntimeError(
                f"Expected persistent {counter}={expected}, but measured {actual}."
            )
    if dimensions != [EXPECTED_EMBEDDING_DIMENSIONS] * expected_queries:
        raise RuntimeError("At least one persistent query embedding dimension changed.")

    return {
        "runs_by_query": runs_by_query,
        "initialization_time": initialization_time,
        "cleanup_time": cleanup_time,
        "timings": dict(timings),
        "counters": dict(counters),
    }


def print_comparison(stateless_runs, persistent):
    print("\n" + "=" * 104)
    print("STATELESS VS PERSISTENT RETRIEVAL")
    print("=" * 104)
    print(
        f"{'Query':<45} {'Stateless cold':>15} {'Stateless warm':>15} "
        f"{'Persistent avg':>16} {'Warm speedup':>14}"
    )
    for query in QUERIES:
        stateless_cold = stateless_runs[query][0]["timings"]["total_retrieval"]
        stateless_warm = stateless_runs[query][1]["timings"]["total_retrieval"]
        persistent_average = mean(
            run["total_retrieval"]
            for run in persistent["runs_by_query"][query]
        )
        speedup = stateless_warm / persistent_average
        print(
            f"{query:<45} {stateless_cold:>15.6f} "
            f"{stateless_warm:>15.6f} {persistent_average:>16.6f} "
            f"{speedup:>13.2f}x"
        )

    all_stateless_warm = [
        stateless_runs[query][1]["timings"]["total_retrieval"]
        for query in QUERIES
    ]
    all_persistent = [
        run["total_retrieval"]
        for query in QUERIES
        for run in persistent["runs_by_query"][query]
    ]
    overall_speedup = mean(all_stateless_warm) / mean(all_persistent)

    print("\nPERSISTENT LIFECYCLE")
    print(f"Initialization: {persistent['initialization_time']:.6f} seconds")
    print(f"Warm query average: {mean(all_persistent):.6f} seconds")
    print(f"Cleanup: {persistent['cleanup_time']:.6f} seconds")
    print(f"Overall warm speedup: {overall_speedup:.2f}x")
    print(
        "Average query embedding generation: "
        f"{mean(run['query_embedding'] for query in QUERIES for run in persistent['runs_by_query'][query]):.6f} seconds"
    )

    print("\nPROOF OF REUSE")
    for name in (
        "database_load_calls",
        "json_deserialization_calls",
        "model_download_calls",
        "model_load_calls",
        "query_embedding_calls",
        "cosine_similarity_calls",
        "model_unload_calls",
    ):
        print(f"{name}: {persistent['counters'].get(name, 0)}")

    print("\nTOP-3 REGRESSION")
    for query in QUERIES:
        print(f"{query}: PASS")
        for rank, result in enumerate(
            persistent["runs_by_query"][query][0]["results"],
            start=1,
        ):
            print(
                f"  {rank}. {result['source']} | chunk {result['chunk_number']} | "
                f"{result['similarity']:.12f}"
            )


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    require_baseline_corpus()
    stateless_runs = run_stateless_comparison()
    persistent = run_persistent_comparison()
    print_comparison(stateless_runs, persistent)

    manager = FoundryLocalManager.instance
    loaded_models = (
        []
        if manager is None
        else [model.id for model in manager.catalog.get_loaded_models()]
    )
    if loaded_models:
        raise RuntimeError(f"Models remain loaded after comparison: {loaded_models}")
    print(f"\nLoaded models after comparison: {loaded_models}")


if __name__ == "__main__":
    main()
