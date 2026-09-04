import tempfile
from pathlib import Path

from fabric_rag.config import EXPECTED_EMBEDDING_DIMENSIONS
from fabric_rag.documents import load_documents, process_documents
from fabric_rag.knowledge_base import rebuild_database
from fabric_rag.management import (
    get_knowledge_base_management_status,
    indexed_chunk_identity,
    remove_source_document,
    save_source_document,
)


def embedded_chunks_for(source_directory):
    documents = load_documents(source_directory)
    processed_documents = process_documents(documents)
    chunks = [
        chunk
        for document in processed_documents
        for chunk in document["chunks"]
    ]
    return [
        {
            **chunk,
            "embedding": [float(index + 1)]
            + [0.0] * (EXPECTED_EMBEDDING_DIMENSIONS - 1),
        }
        for index, chunk in enumerate(chunks)
    ]


def rebuild_fixture(source_directory, database_path):
    rebuild_database(
        embedded_chunks_for(source_directory),
        database_path=database_path,
    )


def raises(expected_exception, operation):
    try:
        operation()
    except expected_exception:
        return True
    return False


def main():
    results = {}

    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        source_directory = root / "documents"
        database_path = root / "rag.db"

        results["A. Safe source upload"] = (
            save_source_document(
                "alpha.md",
                b"# Alpha\n\nA useful first document.",
                source_directory,
            )
            == "alpha.md"
        )
        results["B. Unsupported upload rejected"] = raises(
            ValueError,
            lambda: save_source_document(
                "notes.html",
                b"Not a supported source",
                source_directory,
            ),
        )
        results["C. Duplicate filename rejected"] = raises(
            FileExistsError,
            lambda: save_source_document(
                "alpha.md",
                b"Replacement content",
                source_directory,
            ),
        )
        traversal_target = root / "escape.md"
        results["D. Path traversal rejected"] = (
            raises(
                ValueError,
                lambda: save_source_document(
                    "../escape.md",
                    b"Escape attempt",
                    source_directory,
                ),
            )
            and not traversal_target.exists()
        )
        results["E. Unreadable binary text rejected"] = raises(
            ValueError,
            lambda: save_source_document(
                "invalid.md",
                b"\x00\x01\x02\x03",
                source_directory,
            ),
        )

        save_source_document(
            "beta.md",
            b"# Beta\n\nA useful second document.",
            source_directory,
        )
        rebuild_fixture(source_directory, database_path)
        initial_status = get_knowledge_base_management_status(
            source_directory,
            database_path,
        )
        results["K. Unchanged collection is up to date"] = not initial_status[
            "rebuild_required"
        ]

        save_source_document(
            "gamma.md",
            b"# Gamma\n\nA newly added document.",
            source_directory,
        )
        added_status = get_knowledge_base_management_status(
            source_directory,
            database_path,
        )
        results["H. Added source requires rebuild"] = added_status[
            "rebuild_required"
        ]
        rebuild_fixture(source_directory, database_path)

        (source_directory / "alpha.md").write_text(
            "# Alpha\n\nThe existing document now has changed content.",
            encoding="utf-8",
        )
        modified_status = get_knowledge_base_management_status(
            source_directory,
            database_path,
        )
        results["J. Modified source requires rebuild"] = modified_status[
            "rebuild_required"
        ]
        rebuild_fixture(source_directory, database_path)

        results["F. Safe source removal"] = (
            remove_source_document("beta.md", source_directory) == "beta.md"
            and not (source_directory / "beta.md").exists()
        )
        removed_status = get_knowledge_base_management_status(
            source_directory,
            database_path,
        )
        results["I. Removed source requires rebuild"] = removed_status[
            "rebuild_required"
        ]

        single_source_directory = root / "single-source"
        save_source_document(
            "only.md",
            b"# Only\n\nThe only source document.",
            single_source_directory,
        )
        results["G. Final source removal prevented"] = raises(
            RuntimeError,
            lambda: remove_source_document(
                "only.md",
                single_source_directory,
            ),
        )

        rebuild_fixture(source_directory, database_path)
        identity_before_failure = indexed_chunk_identity(database_path)
        invalid_chunks = embedded_chunks_for(source_directory)
        invalid_chunks[0]["embedding"] = [0.0, 1.0]
        failed_as_expected = raises(
            RuntimeError,
            lambda: rebuild_database(
                invalid_chunks,
                database_path=database_path,
            ),
        )
        identity_after_failure = indexed_chunk_identity(database_path)
        results["L. Failed rebuild preserves valid database"] = (
            failed_as_expected
            and identity_after_failure == identity_before_failure
        )

    print("KNOWLEDGE BASE MANAGER MODEL-FREE VALIDATION")
    print("=" * 48)
    for name, passed in results.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    passed_count = sum(results.values())
    print(f"\nPassed: {passed_count} / {len(results)}")
    if passed_count != len(results):
        raise RuntimeError("At least one knowledge-base management check failed.")


if __name__ == "__main__":
    main()
