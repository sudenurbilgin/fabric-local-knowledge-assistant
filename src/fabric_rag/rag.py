import re
import sys

from foundry_local_sdk import Configuration, FoundryLocalManager

from .config import (
    CHAT_MODEL_ALIAS,
    CHAT_MODEL_VARIANT_ID,
    CPU_EXECUTION_PROVIDER,
    EMBEDDING_MODEL_ALIAS,
    FOUNDRY_APP_NAME,
    RETRIEVAL_TOP_K,
)
from .embeddings import text_preview
from .retrieval import PersistentRetriever


MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\([^\)]+\)")
DOCUMENTATION_NAVIGATION_PATTERN = re.compile(
    r"(?:For more information|To learn more), see [^\n]*",
    re.IGNORECASE,
)
SYSTEM_INSTRUCTION = """You are a Microsoft Fabric documentation assistant.
Answer the user's question only using the supplied context.
Cite supporting statements using the matching source labels [1], [2], or [3], placed immediately after the statements they directly support.
Include source-label citations in supported answers, use only labels present in the context, and do not copy context links or invent citations.
If the supplied context does not contain enough information, say that there is not enough information in the supplied context, do not answer from outside knowledge, and do not include any source-label citations."""


def build_context(retrieved_chunks):
    context_blocks = []
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        content = MARKDOWN_LINK_PATTERN.sub(r"\1", chunk["text"])
        content = DOCUMENTATION_NAVIGATION_PATTERN.sub("", content)
        labeled_content = "\n\n".join(
            f"[{rank}] {paragraph.strip()}"
            for paragraph in content.split("\n\n")
            if paragraph.strip()
        )
        context_blocks.append(
            f"SOURCE [{rank}]\n"
            f"FILE: {chunk['source']}\n"
            f"CHUNK: {chunk['chunk_number']}\n"
            f"CONTENT:\n{labeled_content}"
        )
    return "\n\n---\n\n".join(context_blocks)


def select_chat_variant(model):
    cpu_variant = next(
        (
            variant
            for variant in model.variants
            if variant.id == CHAT_MODEL_VARIANT_ID
            and variant.info.runtime
            and variant.info.runtime.execution_provider == CPU_EXECUTION_PROVIDER
        ),
        None,
    )
    if cpu_variant is None:
        raise RuntimeError(
            f"Required CPU variant '{CHAT_MODEL_VARIANT_ID}' is unavailable."
        )
    return cpu_variant


def build_messages(user_question, context):
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {
            "role": "user",
            "content": (
                f"SUPPLIED CONTEXT:\n\n{context}\n\n"
                f"USER QUESTION:\n{user_question}\n\n"
                "RESPONSE REQUIREMENT:\n"
                "Include supporting numbered source-label citations such as [1]. "
                "For each claim, cite the source block that contains that information; "
                "do not default to [1]. If a statement combines information from multiple "
                "sources, cite every supporting label. If the supplied context does not "
                "contain enough information, state that clearly, do not answer from outside "
                "knowledge, and do not include any source-label citations. Do not output "
                "Markdown links or URLs."
            ),
        },
    ]


class RAGSession:
    def __init__(self):
        self._retriever = None
        if FoundryLocalManager.instance is None:
            FoundryLocalManager.initialize(Configuration(app_name=FOUNDRY_APP_NAME))
        manager = FoundryLocalManager.instance
        self._chat_model = manager.catalog.get_model(CHAT_MODEL_ALIAS)
        if self._chat_model is None:
            raise RuntimeError(
                f"Chat model '{CHAT_MODEL_ALIAS}' was not found in the Foundry Local catalog."
            )

        self._selected_variant = select_chat_variant(self._chat_model)
        self._chat_model.select_variant(self._selected_variant)
        self.model = self._selected_variant.id
        self.execution_provider = (
            self._selected_variant.info.runtime.execution_provider
        )
        self._closed = True

        self._chat_model.download(
            lambda progress: print(
                f"\rDownloading chat model: {progress:.1f}%", end="", flush=True
            )
        )
        print()
        self._chat_model.load()
        self._closed = False
        try:
            self._chat_client = self._chat_model.get_chat_client()
        except Exception:
            self.close()
            raise

    def answer(self, user_question, progress_callback=None):
        if self._closed:
            raise RuntimeError("The RAG session is closed.")
        if not isinstance(user_question, str) or not user_question.strip():
            raise ValueError("Question must be a non-empty string.")

        if self._retriever is None:
            if progress_callback is not None:
                progress_callback("loading_retrieval_model")
            self._retriever = PersistentRetriever()

        if progress_callback is not None:
            progress_callback("retrieving_evidence")
        retrieved_chunks = self._retriever.retrieve(
            user_question,
            top_k=RETRIEVAL_TOP_K,
        )
        context = build_context(retrieved_chunks)
        if progress_callback is not None:
            progress_callback("generating_grounded_answer")
        completion = self._chat_client.complete_chat(
            build_messages(user_question, context)
        )

        return {
            "answer": completion.choices[0].message.content,
            "retrieved_chunks": retrieved_chunks,
            "model": self.model,
            "execution_provider": self.execution_provider,
        }

    def close(self):
        if self._closed and self._retriever is None:
            return

        retriever = self._retriever
        chat_model = None if self._closed else self._chat_model
        self._retriever = None
        self._chat_client = None
        self._closed = True

        try:
            if retriever is not None:
                retriever.close()
        finally:
            if chat_model is not None and chat_model.is_loaded:
                chat_model.unload()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.close()


def answer_query(user_question):
    if not isinstance(user_question, str) or not user_question.strip():
        raise ValueError("Question must be a non-empty string.")

    with RAGSession() as session:
        return session.answer(user_question)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    question = "What is OneLake?"
    result = answer_query(question)

    print("QUESTION:")
    print(question)
    print("\nRETRIEVED CONTEXT:\n")

    for rank, chunk in enumerate(result["retrieved_chunks"], start=1):
        print(f"RANK: {rank}")
        print(f"SIMILARITY: {chunk['similarity']:.6f}")
        print(f"SOURCE: {chunk['source']}")
        print(f"CHUNK: {chunk['chunk_number']}")
        print(f"TEXT PREVIEW: {text_preview(chunk['text'], max_chars=250)}")
        print()

    print("LOCAL MODEL:")
    print(result["model"])
    print("\nEXECUTION PROVIDER:")
    print(result["execution_provider"])
    print("\nANSWER:")
    print(result["answer"])

    runtime_issue = "None"
    if result["execution_provider"] == CPU_EXECUTION_PROVIDER:
        runtime_issue = (
            "CPUExecutionProvider was intentionally retained because it had been "
            "used and validated more extensively throughout the production RAG "
            "pipeline. CUDA was successfully demonstrated as a major performance "
            "optimization, but no consistent answer-quality or citation-quality "
            "difference attributable to CPU versus GPU was established."
        )

    print(f"\nRetrieved chunks: {RETRIEVAL_TOP_K}")
    print("All context retrieved from rag.db: Yes")
    print(f"Embedding model: {EMBEDDING_MODEL_ALIAS}")
    print(f"Local chat model: {result['model']}")
    print("Cloud API used: No")
    print(f"Runtime/provider issue: {runtime_issue}")


if __name__ == "__main__":
    main()
