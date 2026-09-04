import sys

from fabric_rag.rag import RAGSession


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("Microsoft Fabric Documentation Assistant")
    print("Local RAG using Microsoft Foundry Local")
    print("Type 'exit' to quit.")

    with RAGSession() as session:
        while True:
            user_question = input("\nAsk a question: ").strip()

            if user_question.lower() in {"exit", "quit"}:
                print("Goodbye.")
                break

            if not user_question:
                print("Please enter a question.")
                continue

            try:
                result = session.answer(user_question)
            except Exception as error:
                print(f"Error: {error}")
                continue

            print("\nANSWER:")
            print(result["answer"])
            print("\nSOURCES:")
            for rank, chunk in enumerate(result["retrieved_chunks"], start=1):
                print(
                    f"{rank}. {chunk['source']} — chunk {chunk['chunk_number']} "
                    f"— {chunk['similarity']:.6f}"
                )


if __name__ == "__main__":
    main()
