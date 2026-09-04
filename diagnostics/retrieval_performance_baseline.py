import sqlite3
import sys
import time
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from statistics import mean
from unittest.mock import patch

from foundry_local_sdk import FoundryLocalManager

from fabric_rag import retrieval
from fabric_rag.config import (
    DATABASE_PATH,
    EXPECTED_EMBEDDING_DIMENSIONS,
    RETRIEVAL_TOP_K,
)


QUERIES = (
    "What is OneLake?",
    "What is Data Factory in Microsoft Fabric?",
    "How does Direct Lake work?",
    "Who is the CEO of Microsoft?",
)
REPETITIONS = 3
BASELINE_DOCUMENT_COUNT = 8
BASELINE_CHUNK_COUNT = 90
BASELINE_REQUIREMENT_MESSAGE = (
    "This diagnostic requires the original 8-document / 90-chunk baseline corpus."
)
STAGE_ORDER = (
    "manager_initialization_and_orchestration",
    "model_lookup_and_variant_selection",
    "model_download_resolution",
    "model_load",
    "embedding_client_acquisition",
    "query_embedding_generation",
    "model_unload",
    "sqlite_open_read_and_close",
    "embedding_json_deserialization",
    "stored_row_validation_and_orchestration",
    "cosine_similarity",
    "result_assembly_sort_and_top_k",
)


@contextmanager
def measure(timings, stage):
    started = time.perf_counter()
    try:
        yield
    finally:
        timings[stage] += time.perf_counter() - started


class TimedCursor:
    def __init__(self, cursor, timings):
        self._cursor = cursor
        self._timings = timings

    def fetchall(self):
        with measure(self._timings, "sqlite_open_read_and_close"):
            return self._cursor.fetchall()

    def __getattr__(self, name):
        return getattr(self._cursor, name)


class TimedConnection:
    def __init__(self, connection, timings):
        self._connection = connection
        self._timings = timings

    def execute(self, *args, **kwargs):
        with measure(self._timings, "sqlite_open_read_and_close"):
            cursor = self._connection.execute(*args, **kwargs)
        return TimedCursor(cursor, self._timings)

    def close(self):
        with measure(self._timings, "sqlite_open_read_and_close"):
            return self._connection.close()

    def __getattr__(self, name):
        return getattr(self._connection, name)


class TimedSqliteModule:
    def __init__(self, sqlite_module, timings, counters):
        self._sqlite_module = sqlite_module
        self._timings = timings
        self._counters = counters

    def connect(self, *args, **kwargs):
        self._counters["database_open_calls"] += 1
        with measure(self._timings, "sqlite_open_read_and_close"):
            connection = self._sqlite_module.connect(*args, **kwargs)
        return TimedConnection(connection, self._timings)

    def __getattr__(self, name):
        return getattr(self._sqlite_module, name)


class TimedJsonModule:
    def __init__(self, json_module, timings, counters):
        self._json_module = json_module
        self._timings = timings
        self._counters = counters

    def loads(self, *args, **kwargs):
        self._counters["json_deserialization_calls"] += 1
        with measure(self._timings, "embedding_json_deserialization"):
            return self._json_module.loads(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._json_module, name)


class TimedEmbeddingClient:
    def __init__(self, client, timings, counters, dimensions):
        self._client = client
        self._timings = timings
        self._counters = counters
        self._dimensions = dimensions

    def generate_embedding(self, *args, **kwargs):
        self._counters["query_embedding_calls"] += 1
        with measure(self._timings, "query_embedding_generation"):
            response = self._client.generate_embedding(*args, **kwargs)
        self._dimensions.append(len(response.data[0].embedding))
        return response

    def __getattr__(self, name):
        return getattr(self._client, name)


class TimedEmbeddingModel:
    def __init__(self, model, timings, counters, dimensions):
        self._model = model
        self._timings = timings
        self._counters = counters
        self._dimensions = dimensions

    def download(self, *args, **kwargs):
        self._counters["model_download_calls"] += 1
        with measure(self._timings, "model_download_resolution"):
            return self._model.download(*args, **kwargs)

    def load(self, *args, **kwargs):
        self._counters["model_load_calls"] += 1
        with measure(self._timings, "model_load"):
            return self._model.load(*args, **kwargs)

    def get_embedding_client(self, *args, **kwargs):
        with measure(self._timings, "embedding_client_acquisition"):
            client = self._model.get_embedding_client(*args, **kwargs)
        return TimedEmbeddingClient(
            client,
            self._timings,
            self._counters,
            self._dimensions,
        )

    def unload(self, *args, **kwargs):
        self._counters["model_unload_calls"] += 1
        with measure(self._timings, "model_unload"):
            return self._model.unload(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._model, name)


def measure_retrieval(query):
    timings = defaultdict(float)
    counters = defaultdict(int)
    dimensions = []

    original_cosine_similarity = retrieval.cosine_similarity
    original_initialize = retrieval.FoundryLocalManager.initialize
    original_select_embedding_model = retrieval.select_embedding_model
    original_load_stored_chunks = retrieval.load_stored_chunks
    original_generate_query_embedding = retrieval.generate_query_embedding

    def timed_cosine_similarity(*args, **kwargs):
        counters["cosine_similarity_calls"] += 1
        with measure(timings, "cosine_similarity"):
            return original_cosine_similarity(*args, **kwargs)

    def timed_initialize(*args, **kwargs):
        counters["manager_initialization_calls"] += 1
        with measure(timings, "manager_initialization_and_orchestration"):
            return original_initialize(*args, **kwargs)

    def timed_select_embedding_model(manager):
        with measure(timings, "model_lookup_and_variant_selection"):
            model = original_select_embedding_model(manager)
        return TimedEmbeddingModel(model, timings, counters, dimensions)

    def timed_load_stored_chunks():
        started = time.perf_counter()
        try:
            return original_load_stored_chunks()
        finally:
            timings["stored_chunks_total"] += time.perf_counter() - started

    def timed_generate_query_embedding(current_query):
        started = time.perf_counter()
        try:
            return original_generate_query_embedding(current_query)
        finally:
            timings["query_embedding_total"] += time.perf_counter() - started

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                retrieval,
                "sqlite3",
                TimedSqliteModule(retrieval.sqlite3, timings, counters),
            )
        )
        stack.enter_context(
            patch.object(
                retrieval,
                "json",
                TimedJsonModule(retrieval.json, timings, counters),
            )
        )
        stack.enter_context(
            patch.object(retrieval, "cosine_similarity", timed_cosine_similarity)
        )
        stack.enter_context(
            patch.object(
                retrieval.FoundryLocalManager,
                "initialize",
                staticmethod(timed_initialize),
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
                "load_stored_chunks",
                timed_load_stored_chunks,
            )
        )
        stack.enter_context(
            patch.object(
                retrieval,
                "generate_query_embedding",
                timed_generate_query_embedding,
            )
        )

        started = time.perf_counter()
        results = retrieval.get_top_chunks(query, top_k=RETRIEVAL_TOP_K)
        timings["total_retrieval"] = time.perf_counter() - started

    query_measured = sum(
        timings[stage]
        for stage in (
            "manager_initialization_and_orchestration",
            "model_lookup_and_variant_selection",
            "model_download_resolution",
            "model_load",
            "embedding_client_acquisition",
            "query_embedding_generation",
            "model_unload",
        )
    )
    timings["manager_initialization_and_orchestration"] += max(
        0.0,
        timings["query_embedding_total"] - query_measured,
    )

    stored_chunks_measured = (
        timings["sqlite_open_read_and_close"]
        + timings["embedding_json_deserialization"]
    )
    timings["stored_row_validation_and_orchestration"] = max(
        0.0,
        timings["stored_chunks_total"] - stored_chunks_measured,
    )

    timings["result_assembly_sort_and_top_k"] = max(
        0.0,
        timings["total_retrieval"]
        - timings["stored_chunks_total"]
        - timings["query_embedding_total"]
        - timings["cosine_similarity"],
    )

    return {
        "query": query,
        "timings": dict(timings),
        "counters": dict(counters),
        "embedding_dimensions": dimensions,
        "results": results,
    }


def validate_run(run):
    if len(run["results"]) != RETRIEVAL_TOP_K:
        raise RuntimeError(
            f"Expected {RETRIEVAL_TOP_K} results, but received "
            f"{len(run['results'])}."
        )
    if run["embedding_dimensions"] != [EXPECTED_EMBEDDING_DIMENSIONS]:
        raise RuntimeError(
            "The query embedding did not have the expected "
            f"{EXPECTED_EMBEDDING_DIMENSIONS} dimensions."
        )
    expected_counts = {
        "database_open_calls": 1,
        "json_deserialization_calls": BASELINE_CHUNK_COUNT,
        "cosine_similarity_calls": BASELINE_CHUNK_COUNT,
        "model_download_calls": 1,
        "model_load_calls": 1,
        "query_embedding_calls": 1,
        "model_unload_calls": 1,
    }
    for counter, expected in expected_counts.items():
        actual = run["counters"].get(counter, 0)
        if actual != expected:
            raise RuntimeError(
                f"Expected {counter}={expected}, but measured {actual}."
            )


def result_signature(run):
    return [
        (result["source"], result["chunk_number"])
        for result in run["results"]
    ]


def print_individual_run(query, run_number, run):
    print("=" * 78)
    print(f"QUERY: {query}")
    print(f"RUN: {run_number}/{REPETITIONS}")
    print(f"TOTAL RETRIEVAL: {run['timings']['total_retrieval']:.6f} seconds")
    print("STAGES:")
    for stage in STAGE_ORDER:
        print(f"  {stage}: {run['timings'].get(stage, 0.0):.6f} seconds")
    print("TOP-3:")
    for rank, result in enumerate(run["results"], start=1):
        print(
            f"  {rank}. {result['source']} | chunk {result['chunk_number']} | "
            f"similarity {result['similarity']:.12f}"
        )
    print(
        "COUNTS: "
        f"database opens={run['counters'].get('database_open_calls', 0)}, "
        f"JSON embeddings={run['counters'].get('json_deserialization_calls', 0)}, "
        f"cosine comparisons={run['counters'].get('cosine_similarity_calls', 0)}, "
        f"model loads={run['counters'].get('model_load_calls', 0)}, "
        f"model unloads={run['counters'].get('model_unload_calls', 0)}"
    )


def print_summary(runs_by_query):
    print("\n" + "=" * 78)
    print("RETRIEVAL TOTALS BY QUERY")
    print("=" * 78)
    print(
        f"{'Query':<45} {'Run 1':>10} {'Run 2':>10} {'Run 3':>10} "
        f"{'Average':>10} {'Minimum':>10} {'Maximum':>10}"
    )
    for query, runs in runs_by_query.items():
        totals = [run["timings"]["total_retrieval"] for run in runs]
        print(
            f"{query:<45} "
            f"{totals[0]:>10.3f} {totals[1]:>10.3f} {totals[2]:>10.3f} "
            f"{mean(totals):>10.3f} {min(totals):>10.3f} {max(totals):>10.3f}"
        )

    print("\nAVERAGE STAGE TIMINGS (seconds)")
    print(
        f"{'Stage':<46}"
        + "".join(f"{f'Q{index}':>12}" for index in range(1, len(QUERIES) + 1))
        + f"{'All runs':>12}"
    )
    all_runs = [run for runs in runs_by_query.values() for run in runs]
    for stage in STAGE_ORDER:
        per_query = [
            mean(run["timings"].get(stage, 0.0) for run in runs)
            for runs in runs_by_query.values()
        ]
        overall = mean(run["timings"].get(stage, 0.0) for run in all_runs)
        print(
            f"{stage:<46}"
            + "".join(f"{value:>12.6f}" for value in per_query)
            + f"{overall:>12.6f}"
        )

    print("\nFIRST-RUN COMPARISON")
    for query, runs in runs_by_query.items():
        first = runs[0]["timings"]["total_retrieval"]
        later_average = mean(
            run["timings"]["total_retrieval"] for run in runs[1:]
        )
        difference = first - later_average
        percentage = difference / later_average * 100 if later_average else 0.0
        print(
            f"{query}: first={first:.6f}s, later average={later_average:.6f}s, "
            f"difference={difference:+.6f}s ({percentage:+.1f}%)"
        )

    print("\nEXACT TOP-3 BASELINE (RUN 1)")
    for query, runs in runs_by_query.items():
        print(f"\n{query}")
        for rank, result in enumerate(runs[0]["results"], start=1):
            print(
                f"{rank}. source={result['source']} | "
                f"chunk_number={result['chunk_number']} | "
                f"similarity={result['similarity']:.12f}"
            )
        rankings_stable = all(
            result_signature(run) == result_signature(runs[0]) for run in runs[1:]
        )
        print(f"Ranking stable across all runs: {rankings_stable}")


def read_database_counts():
    database_uri = DATABASE_PATH.resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(database_uri, uri=True)
    try:
        return connection.execute(
            "SELECT COUNT(*), COUNT(DISTINCT source) FROM chunks"
        ).fetchone()
    finally:
        connection.close()


def require_baseline_corpus():
    row_count, source_count = read_database_counts()
    if (
        row_count != BASELINE_CHUNK_COUNT
        or source_count != BASELINE_DOCUMENT_COUNT
    ):
        raise RuntimeError(BASELINE_REQUIREMENT_MESSAGE)
    return row_count, source_count


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    require_baseline_corpus()
    runs_by_query = {}
    for query in QUERIES:
        runs = []
        for run_number in range(1, REPETITIONS + 1):
            run = measure_retrieval(query)
            validate_run(run)
            print_individual_run(query, run_number, run)
            runs.append(run)
        runs_by_query[query] = runs

    print_summary(runs_by_query)

    row_count, source_count = read_database_counts()
    if (
        row_count != BASELINE_CHUNK_COUNT
        or source_count != BASELINE_DOCUMENT_COUNT
    ):
        raise RuntimeError(BASELINE_REQUIREMENT_MESSAGE)

    manager = FoundryLocalManager.instance
    loaded_models = (
        []
        if manager is None
        else [model.id for model in manager.catalog.get_loaded_models()]
    )
    if loaded_models:
        raise RuntimeError(f"Models remain loaded after the diagnostic: {loaded_models}")

    print("\n" + "=" * 78)
    print("FINAL VALIDATION")
    print("=" * 78)
    print(f"Database rows: {row_count}")
    print(f"Distinct sources: {source_count}")
    print(f"Embedding dimensions: {EXPECTED_EMBEDDING_DIMENSIONS}")
    print(f"Top-K returned per query: {RETRIEVAL_TOP_K}")
    print(f"Loaded models after diagnostic: {loaded_models}")


if __name__ == "__main__":
    main()
