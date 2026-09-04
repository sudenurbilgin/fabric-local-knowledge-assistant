import json
import subprocess
import sys
import time
from pathlib import Path


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
RESULT_PREFIX = "__PERFORMANCE_RESULT__="
QUESTIONS = [
    "What is OneLake?",
    "What is Data Factory in Microsoft Fabric?",
    "Who is the CEO of Microsoft?",
]


def build_messages(question, context, system_instruction):
    return [
        {"role": "system", "content": system_instruction},
        {
            "role": "user",
            "content": (
                f"SUPPLIED CONTEXT:\n\n{context}\n\n"
                f"USER QUESTION:\n{question}\n\n"
                "RESPONSE REQUIREMENT:\n"
                "Include supporting numbered source-label citations such as [1]. "
                "For each claim, cite the source block that contains that information; "
                "do not default to [1]. If a statement combines information from multiple "
                "sources, cite every supporting label. Do not output Markdown links or URLs."
            ),
        },
    ]


def measure_answer(question):
    from foundry_local_sdk import FoundryLocalManager

    from fabric_rag.rag import (
        CHAT_MODEL_ALIAS,
        SYSTEM_INSTRUCTION,
        build_context,
        select_chat_variant,
    )
    from fabric_rag.retrieval import get_top_chunks

    total_start = time.perf_counter()

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    stage_start = time.perf_counter()
    retrieved_chunks = get_top_chunks(question, top_k=3)
    retrieval_time = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    context = build_context(retrieved_chunks)
    messages = build_messages(question, context, SYSTEM_INSTRUCTION)
    context_time = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    manager = FoundryLocalManager.instance
    chat_model = manager.catalog.get_model(CHAT_MODEL_ALIAS)
    if chat_model is None:
        raise RuntimeError(f"Chat model '{CHAT_MODEL_ALIAS}' was not found.")
    selected_variant = select_chat_variant(chat_model)
    chat_model.select_variant(selected_variant)
    chat_model.download()
    model_prep_time = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    chat_model.load()
    load_time = time.perf_counter() - stage_start

    try:
        stage_start = time.perf_counter()
        chat_client = chat_model.get_chat_client()
        completion = chat_client.complete_chat(messages)
        generation_time = time.perf_counter() - stage_start
        answer = completion.choices[0].message.content
    finally:
        stage_start = time.perf_counter()
        chat_model.unload()
        unload_time = time.perf_counter() - stage_start

    total_time = time.perf_counter() - total_start
    variants = [
        {
            "id": variant.id,
            "execution_provider": (
                variant.info.runtime.execution_provider
                if variant.info.runtime
                else None
            ),
        }
        for variant in chat_model.variants
    ]

    return {
        "question": question,
        "retrieval": retrieval_time,
        "context": context_time,
        "model_prep": model_prep_time,
        "load": load_time,
        "generation": generation_time,
        "unload": unload_time,
        "total": total_time,
        "model": selected_variant.id,
        "execution_provider": selected_variant.info.runtime.execution_provider,
        "variants": variants,
        "answer_received": bool(answer),
    }


def measure_retrieval(question):
    from fabric_rag.retrieval import get_top_chunks

    start = time.perf_counter()
    results = get_top_chunks(question, top_k=3)
    elapsed = time.perf_counter() - start
    return {
        "question": question,
        "retrieval_only": elapsed,
        "result_count": len(results),
    }


def run_child(mode, question):
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "diagnostics.test_performance",
            mode,
            question,
        ],
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
        raise RuntimeError("Performance subprocess returned no structured result.")
    return json.loads(result_line[len(RESULT_PREFIX) :])


def print_report(answer_results, retrieval_results):
    print("END-TO-END STAGE TIMINGS (seconds)")
    print(
        f"{'Question':<43} {'Retrieval':>10} {'Context':>9} {'Prep':>9} "
        f"{'Load':>9} {'Generate':>10} {'Unload':>9} {'Total':>9}"
    )
    print("-" * 114)
    for result in answer_results:
        print(
            f"{result['question']:<43} {result['retrieval']:>10.3f} "
            f"{result['context']:>9.3f} {result['model_prep']:>9.3f} "
            f"{result['load']:>9.3f} {result['generation']:>10.3f} "
            f"{result['unload']:>9.3f} {result['total']:>9.3f}"
        )

    print("\nRETRIEVAL-ONLY TIMINGS (query embedding + cosine search)")
    for result in retrieval_results:
        print(f"{result['question']}: {result['retrieval_only']:.3f} seconds")

    first = answer_results[0]
    print(f"\nExact model variant: {first['model']}")
    print(f"Selected execution provider: {first['execution_provider']}")
    print("Exposed phi-4-mini variants:")
    for variant in first["variants"]:
        print(f"- {variant['id']} — {variant['execution_provider']}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if len(sys.argv) == 3 and sys.argv[1] in {"--answer", "--retrieval"}:
        result = (
            measure_answer(sys.argv[2])
            if sys.argv[1] == "--answer"
            else measure_retrieval(sys.argv[2])
        )
        print(RESULT_PREFIX + json.dumps(result, ensure_ascii=False))
        return

    answer_results = [run_child("--answer", question) for question in QUESTIONS]
    retrieval_results = [
        run_child("--retrieval", question) for question in QUESTIONS
    ]
    print_report(answer_results, retrieval_results)


if __name__ == "__main__":
    main()
