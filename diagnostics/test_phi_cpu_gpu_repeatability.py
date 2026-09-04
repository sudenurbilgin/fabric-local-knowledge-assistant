import re
import subprocess
import sys
import threading
import time

from foundry_local_sdk import Configuration, FoundryLocalManager

from fabric_rag.rag import build_context, build_messages
from fabric_rag.retrieval import get_top_chunks


CPU_VARIANT_ID = "Phi-4-mini-instruct-generic-cpu:5"
CUDA_VARIANT_ID = "Phi-4-mini-instruct-cuda-gpu:5"
CITATION_QUESTIONS = [
    "What is Data Factory in Microsoft Fabric?",
    "How does Direct Lake work?",
]
UNSUPPORTED_QUESTION = "Who is the CEO of Microsoft?"
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


def complete_chat(client, messages, monitor_cuda=False):
    samples = []
    stop_event = threading.Event()
    monitor = None
    if monitor_cuda:
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
        if monitor is not None:
            stop_event.set()
            monitor.join(timeout=2)

    answer = completion.choices[0].message.content or ""
    return {
        "answer": answer,
        "citations": CITATION_PATTERN.findall(answer),
        "generation_time": elapsed,
        "max_gpu_memory": max((sample[0] for sample in samples), default=None),
        "max_gpu_utilization": max(
            (sample[1] for sample in samples), default=None
        ),
    }


def retrieve_inputs_once():
    inputs = {}
    for question in [*CITATION_QUESTIONS, UNSUPPORTED_QUESTION]:
        chunks = get_top_chunks(question, top_k=3)
        inputs[question] = {
            "chunks": chunks,
            "messages": build_messages(question, build_context(chunks)),
        }
    return inputs


def print_shared_context(inputs):
    print("SHARED RETRIEVED CONTEXTS")
    for question, data in inputs.items():
        print("\n" + "=" * 72)
        print(f"QUESTION: {question}")
        for rank, chunk in enumerate(data["chunks"], start=1):
            print("\n" + "-" * 72)
            print(
                f"RANK {rank} | {chunk['source']} | chunk "
                f"{chunk['chunk_number']} | {chunk['similarity']:.6f}"
            )
            print(chunk["text"])


def get_exact_variant(manager, variant_id):
    model = manager.catalog.get_model("phi-4-mini")
    if model is None:
        raise RuntimeError("The phi-4-mini catalog entry is unavailable.")
    variant = next(
        (candidate for candidate in model.variants if candidate.id == variant_id),
        None,
    )
    if variant is None:
        raise RuntimeError(f"Variant '{variant_id}' is unavailable.")
    model.select_variant(variant)
    return model, variant


def run_citation_repetitions(manager, variant_id, provider, inputs):
    model, variant = get_exact_variant(manager, variant_id)
    model.download()
    model.load()
    results = []
    try:
        client = model.get_chat_client()
        for question in CITATION_QUESTIONS:
            for repetition in range(1, 4):
                result = complete_chat(
                    client,
                    inputs[question]["messages"],
                    monitor_cuda=provider == "CUDAExecutionProvider",
                )
                record = {
                    "question": question,
                    "provider": provider,
                    "variant": variant.id,
                    "repetition": repetition,
                    **result,
                }
                results.append(record)
                print_run(record)
    finally:
        model.unload()
    return results


def run_unsupported_check(manager, variant_id, provider, inputs):
    model, variant = get_exact_variant(manager, variant_id)
    model.download()
    model.load()
    try:
        client = model.get_chat_client()
        result = complete_chat(
            client,
            inputs[UNSUPPORTED_QUESTION]["messages"],
            monitor_cuda=provider == "CUDAExecutionProvider",
        )
        record = {
            "question": UNSUPPORTED_QUESTION,
            "provider": provider,
            "variant": variant.id,
            "repetition": 1,
            **result,
        }
        print_run(record)
        return record
    finally:
        model.unload()


def print_run(record):
    print("\n" + "=" * 72)
    print(f"QUESTION: {record['question']}")
    print(f"PROVIDER: {record['provider']}")
    print(f"VARIANT: {record['variant']}")
    print(f"REPETITION: {record['repetition']}")
    print(f"GENERATION TIME: {record['generation_time']:.3f} seconds")
    print(
        "CITATIONS: "
        + (
            ", ".join(f"[{label}]" for label in record["citations"])
            or "None"
        )
    )
    if record["provider"] == "CUDAExecutionProvider":
        print(f"MAX GPU VRAM: {record['max_gpu_memory']} MiB")
        print(f"MAX GPU UTILIZATION: {record['max_gpu_utilization']}%")
    print("ANSWER:")
    print(record["answer"])


def format_citations(record):
    return ", ".join(f"[{label}]" for label in record["citations"]) or "None"


def print_question_summary(title, question, cpu_results, cuda_results):
    print(title)
    for provider_name, results in (("CPU", cpu_results), ("CUDA", cuda_results)):
        print(f"{provider_name}:")
        question_results = [
            record for record in results if record["question"] == question
        ]
        for record in question_results:
            print(
                f"Run {record['repetition']} citations: "
                f"{format_citations(record)}"
            )
        print()


def print_final_summary(
    cpu_results,
    cuda_results,
    cpu_unsupported,
    cuda_unsupported,
    loaded_models,
):
    cpu_average = sum(
        item["generation_time"] for item in cpu_results
    ) / len(cpu_results)
    cuda_average = sum(
        item["generation_time"] for item in cuda_results
    ) / len(cuda_results)
    cuda_memory = max(
        item["max_gpu_memory"]
        for item in [*cuda_results, cuda_unsupported]
        if item["max_gpu_memory"] is not None
    )
    cuda_utilization = max(
        item["max_gpu_utilization"]
        for item in [*cuda_results, cuda_unsupported]
        if item["max_gpu_utilization"] is not None
    )

    print("\n" + "=" * 72)
    print("FINAL CPU VS CUDA REPEATABILITY SUMMARY")
    print("=" * 72)
    print_question_summary(
        "DATA FACTORY",
        CITATION_QUESTIONS[0],
        cpu_results,
        cuda_results,
    )
    print_question_summary(
        "DIRECT LAKE",
        CITATION_QUESTIONS[1],
        cpu_results,
        cuda_results,
    )
    print("PERFORMANCE")
    print(f"CPU average generation time: {cpu_average:.3f} seconds")
    print(f"CUDA average generation time: {cuda_average:.3f} seconds")
    print(f"CUDA speedup: {cpu_average / cuda_average:.2f}x")
    print("\nUNSUPPORTED QUERY")
    print(f"CPU answer: {cpu_unsupported['answer']}")
    print(f"CUDA answer: {cuda_unsupported['answer']}")
    print("\nGPU EXECUTION")
    print(f"CUDA variant: {cuda_results[0]['variant']}")
    print(f"Execution provider: {cuda_results[0]['provider']}")
    print(f"Maximum observed GPU VRAM: {cuda_memory} MiB")
    print(f"Maximum observed GPU utilization: {cuda_utilization} %")
    print("\nCLEANUP")
    print(f"Loaded models after diagnostic: {loaded_models}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if FoundryLocalManager.instance is None:
        FoundryLocalManager.initialize(Configuration(app_name="foundry_local_rag"))
    manager = FoundryLocalManager.instance
    registration = manager.download_and_register_eps(
        names=["CUDAExecutionProvider"]
    )
    print(f"CUDA REGISTRATION: {registration}")

    inputs = retrieve_inputs_once()
    print_shared_context(inputs)

    print("\n" + "#" * 72)
    print("CPU CITATION REPEATABILITY")
    cpu_results = run_citation_repetitions(
        manager,
        CPU_VARIANT_ID,
        "CPUExecutionProvider",
        inputs,
    )

    print("\n" + "#" * 72)
    print("CUDA CITATION REPEATABILITY")
    cuda_results = run_citation_repetitions(
        manager,
        CUDA_VARIANT_ID,
        "CUDAExecutionProvider",
        inputs,
    )

    print("\n" + "#" * 72)
    print("FINAL UNSUPPORTED CHECK")
    cpu_unsupported = run_unsupported_check(
        manager,
        CPU_VARIANT_ID,
        "CPUExecutionProvider",
        inputs,
    )
    cuda_unsupported = run_unsupported_check(
        manager,
        CUDA_VARIANT_ID,
        "CUDAExecutionProvider",
        inputs,
    )

    print("\n" + "#" * 72)
    print("TIMING SUMMARY")
    print(
        "CPU CITATION-RUN AVERAGE: "
        f"{sum(item['generation_time'] for item in cpu_results) / len(cpu_results):.3f} seconds"
    )
    print(
        "CUDA CITATION-RUN AVERAGE: "
        f"{sum(item['generation_time'] for item in cuda_results) / len(cuda_results):.3f} seconds"
    )
    print(
        "CPU UNSUPPORTED TIME: "
        f"{cpu_unsupported['generation_time']:.3f} seconds"
    )
    print(
        "CUDA UNSUPPORTED TIME: "
        f"{cuda_unsupported['generation_time']:.3f} seconds"
    )
    loaded_models = [model.id for model in manager.catalog.get_loaded_models()]
    print(f"LOADED MODELS AFTER DIAGNOSTIC: {loaded_models}")

    print_final_summary(
        cpu_results,
        cuda_results,
        cpu_unsupported,
        cuda_unsupported,
        loaded_models,
    )


if __name__ == "__main__":
    main()
