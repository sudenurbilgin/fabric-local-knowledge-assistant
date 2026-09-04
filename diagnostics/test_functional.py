import json
import re
import subprocess
import sys
from pathlib import Path


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
RESULT_PREFIX = "__FUNCTIONAL_RESULT__="
CITATION_PATTERN = re.compile(r"\[([1-3])\]")

TEST_CASES = [
    ("SUPPORTED", "What is Microsoft Fabric?"),
    ("SUPPORTED", "What is OneLake?"),
    ("SUPPORTED", "What is a lakehouse in Microsoft Fabric?"),
    ("SUPPORTED", "What is Data Factory in Microsoft Fabric?"),
    ("SUPPORTED", "How does Direct Lake work?"),
    ("UNSUPPORTED", "Who is the CEO of Microsoft?"),
    ("UNSUPPORTED", "What is the weather in Istanbul today?"),
    ("GENERAL / BROAD", "Tell me about Microsoft Fabric."),
]

CHILD_CODE = """
import json
import sys

from fabric_rag.rag import answer_query

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

result = answer_query(sys.argv[1])
print("__FUNCTIONAL_RESULT__=" + json.dumps(result, ensure_ascii=False))
"""


def run_question(question):
    completed = subprocess.run(
        [sys.executable, "-B", "-c", CHILD_CODE, question],
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
        raise RuntimeError("The answer subprocess did not return structured results.")
    return json.loads(result_line[len(RESULT_PREFIX) :])


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    for test_number, (category, question) in enumerate(TEST_CASES, start=1):
        result = run_question(question)
        citations = CITATION_PATTERN.findall(result["answer"])

        print("=" * 70)
        print(f"TEST {test_number}")
        print(f"CATEGORY: {category}")
        print(f"QUESTION: {question}")
        print("TOP-3 SOURCES:")
        for rank, chunk in enumerate(result["retrieved_chunks"], start=1):
            print(
                f"{rank}. {chunk['source']} — chunk {chunk['chunk_number']} "
                f"— {chunk['similarity']:.6f}"
            )
        print("ANSWER:")
        print(result["answer"])
        print("CITATION LABELS USED: " + (", ".join(f"[{c}]" for c in citations) or "None"))
        print()


if __name__ == "__main__":
    main()
