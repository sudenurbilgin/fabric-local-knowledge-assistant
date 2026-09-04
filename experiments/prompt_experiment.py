import json
from pathlib import Path
from urllib.parse import urlparse

from openai import OpenAI


SYSTEM_MESSAGE = """You are a documentation assistant.
Answer the user's question using ONLY the provided context.
Do not add facts that are not present in the context.
If the context does not contain enough information, say:
'I do not have enough information in the provided context.'"""

CONTEXT = (
    "Direct Lake is a storage mode for Power BI semantic models in Microsoft Fabric. "
    "It allows Power BI to access data stored in OneLake directly, without importing "
    "the data into the semantic model."
)


def get_daemon_url():
    metadata_path = Path.home() / ".foundry" / "daemon.json"
    with metadata_path.open(encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    for url in metadata["web_urls"]:
        if urlparse(url).hostname in {"127.0.0.1", "localhost", "::1"}:
            return url.rstrip("/")

    raise RuntimeError("The Foundry Local daemon has no localhost endpoint.")


def get_model_name(client):
    for model in client.models.list().data:
        if model.id.startswith("qwen2.5-0.5b"):
            return model.id

    raise RuntimeError("The local qwen2.5-0.5b model is not exposed by the daemon.")


def generate_response(client, model_name, messages):
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
    )
    return completion.choices[0].message.content


def print_experiment(title, question, response):
    print("=" * 60)
    print(title)
    print("=" * 60)
    print("Question:")
    print(question)
    print()
    print("Response:")
    print(response)
    print()


def main():
    daemon_url = get_daemon_url()
    client = OpenAI(base_url=f"{daemon_url}/v1", api_key="local")
    model_name = get_model_name(client)

    question_1 = "What is Direct Lake in Microsoft Fabric?"
    response_1 = generate_response(
        client,
        model_name,
        [{"role": "user", "content": question_1}],
    )

    question_2 = "What is Direct Lake?"
    response_2 = generate_response(
        client,
        model_name,
        [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": CONTEXT},
            {"role": "user", "content": question_2},
        ],
    )

    question_3 = "Who created Microsoft Fabric?"
    response_3 = generate_response(
        client,
        model_name,
        [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": CONTEXT},
            {"role": "user", "content": question_3},
        ],
    )

    print_experiment("EXPERIMENT 1 — NO CONTEXT", question_1, response_1)
    print_experiment("EXPERIMENT 2 — GROUNDED CONTEXT", question_2, response_2)
    print_experiment("EXPERIMENT 3 — MISSING INFORMATION", question_3, response_3)

    print(f"Model: {model_name}")
    print("Inference: Local — Microsoft Foundry Local")


if __name__ == "__main__":
    main()
