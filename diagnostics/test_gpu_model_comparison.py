import re
import subprocess
import sys
import threading
import time

from foundry_local_sdk import Configuration, FoundryLocalManager

from fabric_rag.rag import build_context, build_messages
from fabric_rag.retrieval import get_top_chunks


QUESTIONS = [
    "What is OneLake?",
    "What is Data Factory in Microsoft Fabric?",
    "How does Direct Lake work?",
    "Who is the CEO of Microsoft?",
]
MODEL_VARIANTS = [
    ("phi-4-mini", "Phi-4-mini-instruct-generic-cpu:5"),
    ("phi-4-mini", "Phi-4-mini-instruct-cuda-gpu:5"),
    (
        "ministral-3-3b-instruct-2512",
        "ministral-3-3b-instruct-2512-cuda-gpu:1",
    ),
]
CITATION_PATTERN = re.compile(r"\[([1-3])\]")


def read_gpu_metrics():
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        encoding="utf-8",
    ).strip()
    memory, utilization = output.split(",", maxsplit=1)
    return int(memory.strip()), int(utilization.strip())


def monitor_gpu(stop_event, samples):
    while not stop_event.is_set():
        try:
            samples.append(read_gpu_metrics())
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        stop_event.wait(0.5)


def generate_with_metrics(client, messages):
    samples = []
    stop_event = threading.Event()
    monitor = threading.Thread(
        target=monitor_gpu,
        args=(stop_event, samples),
        daemon=True,
    )
    monitor.start()
    start = time.perf_counter()
    try:
        completion = client.complete_chat(messages)
    finally:
        elapsed = time.perf_counter() - start
        stop_event.set()
        monitor.join(timeout=2)

    answer = completion.choices[0].message.content or ""
    return {
        "answer": answer,
        "generation_time": elapsed,
        "citations": CITATION_PATTERN.findall(answer),
        "gpu_samples": samples,
    }


def retrieve_shared_inputs():
    shared_inputs = {}
    for question in QUESTIONS:
        retrieved_chunks = get_top_chunks(question, top_k=3)
        shared_inputs[question] = {
            "retrieved_chunks": retrieved_chunks,
            "messages": build_messages(question, build_context(retrieved_chunks)),
        }
    return shared_inputs


def select_exact_variant(model, variant_id):
    variant = next(
        (variant for variant in model.variants if variant.id == variant_id),
        None,
    )
    if variant is None:
        raise RuntimeError(f"Variant '{variant_id}' is not exposed.")
    model.select_variant(variant)
    return variant


def test_variant(manager, alias, variant_id, shared_inputs):
    model = manager.catalog.get_model(alias)
    if model is None:
        raise RuntimeError(f"Model alias '{alias}' is not exposed.")
    variant = select_exact_variant(model, variant_id)

    download_start = time.perf_counter()
    progress_milestones = set()

    def report_download_progress(progress):
        milestone = min(100, int(progress // 25) * 25)
        if milestone not in progress_milestones:
            progress_milestones.add(milestone)
            print(f"Downloading {variant_id}: {milestone}%", flush=True)

    model.download(report_download_progress)
    download_time = time.perf_counter() - download_start

    load_start = time.perf_counter()
    model.load()
    load_time = time.perf_counter() - load_start

    results = []
    try:
        client = model.get_chat_client()
        for question in QUESTIONS:
            results.append(
                {
                    "question": question,
                    **generate_with_metrics(
                        client,
                        shared_inputs[question]["messages"],
                    ),
                }
            )
    finally:
        model.unload()

    return {
        "alias": alias,
        "variant": variant.id,
        "execution_provider": variant.info.runtime.execution_provider,
        "device_type": str(variant.info.runtime.device_type),
        "download_time": download_time,
        "load_time": load_time,
        "results": results,
    }


def print_shared_inputs(shared_inputs):
    print("SHARED RETRIEVAL INPUTS")
    for question in QUESTIONS:
        print(f"\nQUESTION: {question}")
        for rank, chunk in enumerate(
            shared_inputs[question]["retrieved_chunks"], start=1
        ):
            print(
                f"{rank}. {chunk['source']} — chunk {chunk['chunk_number']} "
                f"— {chunk['similarity']:.6f}"
            )


def print_variant_result(result):
    print("\n" + "=" * 72)
    print(f"ALIAS: {result['alias']}")
    print(f"VARIANT: {result['variant']}")
    print(f"EXECUTION PROVIDER: {result['execution_provider']}")
    print(f"DEVICE TYPE: {result['device_type']}")
    print(f"DOWNLOAD/CACHE CHECK: {result['download_time']:.3f} seconds")
    print(f"MODEL LOAD: {result['load_time']:.3f} seconds")

    for question_result in result["results"]:
        samples = question_result["gpu_samples"]
        max_memory = max((sample[0] for sample in samples), default=None)
        max_utilization = max((sample[1] for sample in samples), default=None)
        print("\nQUESTION:")
        print(question_result["question"])
        print(f"GENERATION TIME: {question_result['generation_time']:.3f} seconds")
        print(f"MAX OBSERVED GPU MEMORY: {max_memory} MiB")
        print(f"MAX OBSERVED GPU UTILIZATION: {max_utilization}%")
        print("ANSWER:")
        print(question_result["answer"])
        print(
            "CITATIONS: "
            + (
                ", ".join(
                    f"[{citation}]" for citation in question_result["citations"]
                )
                or "None"
            )
        )


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if FoundryLocalManager.instance is None:
        FoundryLocalManager.initialize(Configuration(app_name="foundry_local_rag"))
    manager = FoundryLocalManager.instance
    registration = manager.download_and_register_eps()
    print(f"EP REGISTRATION: {registration}")

    shared_inputs = retrieve_shared_inputs()
    print_shared_inputs(shared_inputs)

    for alias, variant_id in MODEL_VARIANTS:
        try:
            print_variant_result(
                test_variant(manager, alias, variant_id, shared_inputs)
            )
        except Exception as error:
            print("\n" + "=" * 72)
            print(f"VARIANT: {variant_id}")
            print(f"RUNTIME ERROR: {type(error).__name__}: {error}")

    print(
        "\nLOADED MODELS AFTER COMPARISON: "
        f"{[model.id for model in manager.catalog.get_loaded_models()]}"
    )


if __name__ == "__main__":
    main()
