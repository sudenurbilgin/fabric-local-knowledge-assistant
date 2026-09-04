import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from fabric_rag.config import CHUNK_SIZE, EXPECTED_EMBEDDING_DIMENSIONS
from fabric_rag.documents import load_documents, process_documents
from fabric_rag.knowledge_base import (
    get_knowledge_base_statistics,
    rebuild_database,
)


def flatten_chunks(processed_documents):
    return [
        chunk
        for document in processed_documents
        for chunk in document["chunks"]
    ]


def add_fixture_document(directory, filename, title, body):
    path = directory / filename
    path.write_text(f"# {title}\n\n{body}\n", encoding="utf-8")
    return path


def add_fixture_embeddings(chunks):
    return [
        {
            **chunk,
            "embedding": [float(index + 1)]
            + [0.0] * (EXPECTED_EMBEDDING_DIMENSIONS - 1),
        }
        for index, chunk in enumerate(chunks)
    ]


def assert_valid_chunks(chunks):
    if not chunks:
        raise RuntimeError("The temporary corpus produced no chunks.")
    if any(not chunk["source"].strip() for chunk in chunks):
        raise RuntimeError("A temporary chunk has no source filename.")
    if any(chunk["chunk_number"] <= 0 for chunk in chunks):
        raise RuntimeError("A temporary chunk has an invalid chunk number.")
    if any(not chunk["text"].strip() for chunk in chunks):
        raise RuntimeError("A temporary chunk is empty.")
    if max(len(chunk["text"]) for chunk in chunks) > CHUNK_SIZE:
        raise RuntimeError("A temporary chunk exceeds the production size limit.")
    keys = [
        (chunk["source"], chunk["chunk_number"])
        for chunk in chunks
    ]
    if len(keys) != len(set(keys)):
        raise RuntimeError("The temporary corpus contains duplicate chunk keys.")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    with TemporaryDirectory(prefix="fabric-rag-dynamic-") as temporary_root:
        root = Path(temporary_root)
        document_directory = root / "documents"
        document_directory.mkdir()
        database_path = root / "dynamic-rag.db"

        empty_directory = root / "empty-documents"
        empty_directory.mkdir()
        try:
            load_documents(empty_directory)
        except RuntimeError as error:
            empty_collection_rejected = "No supported source documents" in str(error)
        else:
            empty_collection_rejected = False
        if not empty_collection_rejected:
            raise RuntimeError("An empty supported-source collection was not rejected.")

        add_fixture_document(
            document_directory,
            "zeta.md",
            "Zeta",
            "A short Markdown document used to validate dynamic discovery.",
        )
        add_fixture_document(
            document_directory,
            "alpha.md",
            "Alpha",
            "Another document used by the real production chunking functions.",
        )
        (document_directory / "ignored.html").write_text(
            "This unsupported file type must be ignored.",
            encoding="utf-8",
        )

        initial_documents = load_documents(document_directory)
        initial_sources = [document["source"] for document in initial_documents]
        if initial_sources != ["alpha.md", "zeta.md"]:
            raise RuntimeError(
                f"Source discovery was not deterministic: {initial_sources}"
            )
        initial_chunks = flatten_chunks(process_documents(initial_documents))
        assert_valid_chunks(initial_chunks)
        rebuild_database(
            add_fixture_embeddings(initial_chunks),
            database_path=database_path,
        )
        initial_statistics = get_knowledge_base_statistics(database_path)
        if initial_statistics != {
            "document_count": len(initial_documents),
            "chunk_count": len(initial_chunks),
        }:
            raise RuntimeError(
                f"Initial dynamic statistics are incorrect: {initial_statistics}"
            )

        add_fixture_document(
            document_directory,
            "middle.md",
            "Middle",
            "Adding a Markdown document must change the discovered corpus naturally.",
        )
        expanded_documents = load_documents(document_directory)
        expanded_chunks = flatten_chunks(process_documents(expanded_documents))
        assert_valid_chunks(expanded_chunks)
        if len(expanded_documents) != len(initial_documents) + 1:
            raise RuntimeError("The additional source document was not discovered.")
        if len(expanded_chunks) <= len(initial_chunks):
            raise RuntimeError("The additional document did not increase the chunk count.")

        valid_embedded_chunks = add_fixture_embeddings(expanded_chunks)
        rebuild_database(valid_embedded_chunks, database_path=database_path)
        expanded_statistics = get_knowledge_base_statistics(database_path)
        expected_expanded_statistics = {
            "document_count": len(expanded_documents),
            "chunk_count": len(expanded_chunks),
        }
        if expanded_statistics != expected_expanded_statistics:
            raise RuntimeError(
                f"Expanded dynamic statistics are incorrect: {expanded_statistics}"
            )

        invalid_embedded_chunks = [
            {**chunk, "embedding": list(chunk["embedding"])}
            for chunk in valid_embedded_chunks
        ]
        invalid_embedded_chunks[-1]["embedding"] = [0.0]
        try:
            rebuild_database(
                invalid_embedded_chunks,
                database_path=database_path,
            )
        except RuntimeError:
            failed_rebuild_rolled_back = (
                get_knowledge_base_statistics(database_path)
                == expanded_statistics
            )
        else:
            failed_rebuild_rolled_back = False
        if not failed_rebuild_rolled_back:
            raise RuntimeError("A failed rebuild did not preserve the previous database.")

        real_statistics = get_knowledge_base_statistics()

        print("DYNAMIC CORPUS VALIDATION")
        print(f"Empty supported-source collection rejected: {empty_collection_rejected}")
        print(f"Unsupported file ignored: {'ignored.html' not in initial_sources}")
        print(f"Initial documents: {len(initial_documents)}")
        print(f"Initial chunks: {len(initial_chunks)}")
        print(f"Expanded documents: {len(expanded_documents)}")
        print(f"Expanded chunks: {len(expanded_chunks)}")
        print(f"Maximum temporary chunk length: {max(len(chunk['text']) for chunk in expanded_chunks)}")
        print(f"Temporary database statistics: {expanded_statistics}")
        print(f"Failed rebuild preserved prior database: {failed_rebuild_rolled_back}")
        print(f"Current production database statistics: {real_statistics}")
        print("Real data/documents modified: False")
        print("Real rag.db modified: False")
        print("Model inference used: False")


if __name__ == "__main__":
    main()
