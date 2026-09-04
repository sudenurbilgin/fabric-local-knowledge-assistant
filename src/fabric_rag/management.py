import sqlite3
from pathlib import Path

from .config import DATABASE_PATH, DOCUMENTS_DIRECTORY
from .documents import load_documents, process_documents
from .ingestion import (
    SUPPORTED_SOURCE_SUFFIXES,
    discover_source_paths,
    extract_source_text,
)


INVALID_FILENAME_CHARACTERS = frozenset('<>:"/\\|?*')
WINDOWS_RESERVED_NAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{number}" for number in range(1, 10)),
        *(f"LPT{number}" for number in range(1, 10)),
    }
)


def validate_source_filename(filename):
    if not isinstance(filename, str) or not filename:
        raise ValueError("The uploaded file must have a filename.")
    if filename != filename.strip() or filename.endswith((".", " ")):
        raise ValueError("The filename cannot begin or end with spaces or dots.")
    if Path(filename).name != filename or any(
        character in INVALID_FILENAME_CHARACTERS for character in filename
    ):
        raise ValueError("The filename must not contain a path or unsafe characters.")
    if any(ord(character) < 32 for character in filename):
        raise ValueError("The filename contains an unsupported control character.")
    if Path(filename).suffix.casefold() not in SUPPORTED_SOURCE_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SOURCE_SUFFIXES))
        raise ValueError(f"Unsupported source type. Supported types: {supported}.")
    if not Path(filename).stem:
        raise ValueError("The source filename must include a name before its extension.")
    if Path(filename).stem.split(".", maxsplit=1)[0].upper() in WINDOWS_RESERVED_NAMES:
        raise ValueError("That filename is reserved by Windows.")

    return filename


def source_path(filename, source_directory=DOCUMENTS_DIRECTORY):
    safe_filename = validate_source_filename(filename)
    directory = source_directory.resolve()
    target = (directory / safe_filename).resolve()
    if target.parent != directory:
        raise ValueError("The source file must remain inside the documents directory.")
    return target


def list_source_files(source_directory=DOCUMENTS_DIRECTORY):
    return [path.name for path in discover_source_paths(source_directory)]


def get_source_file_details(source_directory=DOCUMENTS_DIRECTORY):
    return [
        {
            "filename": path.name,
            "file_type": path.suffix.casefold().removeprefix(".").upper(),
            "size_bytes": path.stat().st_size,
        }
        for path in discover_source_paths(source_directory)
    ]


def save_source_document(
    filename,
    content,
    source_directory=DOCUMENTS_DIRECTORY,
):
    validate_source_filename(filename)
    source_directory.mkdir(parents=True, exist_ok=True)
    target = source_path(filename, source_directory)
    if target.exists():
        raise FileExistsError(
            f"A source document named '{filename}' already exists."
        )

    extract_source_text(filename, content)
    try:
        with target.open("xb") as destination:
            destination.write(content)
    except FileExistsError as error:
        raise FileExistsError(
            f"A source document named '{filename}' already exists."
        ) from error
    except Exception:
        if target.exists():
            target.unlink()
        raise

    return target.name


def remove_source_document(filename, source_directory=DOCUMENTS_DIRECTORY):
    target = source_path(filename, source_directory)
    current_files = list_source_files(source_directory)
    if target.name not in current_files:
        raise FileNotFoundError(f"Source document '{filename}' was not found.")
    if len(current_files) <= 1:
        raise RuntimeError("At least one usable source document must remain.")

    remaining_paths = [
        source_directory / current_filename
        for current_filename in current_files
        if current_filename != target.name
    ]
    if not any(
        _source_is_usable(path)
        for path in remaining_paths
    ):
        raise RuntimeError("At least one usable source document must remain.")

    target.unlink()
    return target.name


def _source_is_usable(path):
    try:
        extract_source_text(path.name, path.read_bytes())
    except (OSError, TypeError, ValueError):
        return False
    return True


def validate_markdown_filename(filename):
    validated = validate_source_filename(filename)
    if Path(validated).suffix.casefold() != ".md":
        raise ValueError("Only Markdown (.md) files are supported by this legacy helper.")
    return validated


def list_markdown_source_files(source_directory=DOCUMENTS_DIRECTORY):
    return [
        filename
        for filename in list_source_files(source_directory)
        if Path(filename).suffix.casefold() == ".md"
    ]


def save_markdown_source(filename, content, source_directory=DOCUMENTS_DIRECTORY):
    validate_markdown_filename(filename)
    return save_source_document(filename, content, source_directory)


def remove_markdown_source(filename, source_directory=DOCUMENTS_DIRECTORY):
    validate_markdown_filename(filename)
    return remove_source_document(filename, source_directory)


def current_processed_chunk_identity(source_directory=DOCUMENTS_DIRECTORY):
    documents = load_documents(source_directory)
    processed_documents = process_documents(documents)
    return sorted(
        (
            chunk["source"],
            chunk["chunk_number"],
            chunk["text"],
        )
        for document in processed_documents
        for chunk in document["chunks"]
    )


def indexed_chunk_identity(database_path=DATABASE_PATH):
    if not database_path.is_file():
        return []

    database_uri = database_path.resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(database_uri, uri=True)
    try:
        rows = connection.execute(
            """
            SELECT source, chunk_number, text
            FROM chunks
            ORDER BY source, chunk_number
            """
        ).fetchall()
    finally:
        connection.close()
    return rows


def get_indexed_passage_counts_by_source(database_path=DATABASE_PATH):
    if not database_path.is_file():
        return []

    database_uri = database_path.resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(database_uri, uri=True)
    try:
        return connection.execute(
            """
            SELECT source, COUNT(*)
            FROM chunks
            GROUP BY source
            ORDER BY source
            """
        ).fetchall()
    finally:
        connection.close()


def get_knowledge_base_management_status(
    source_directory=DOCUMENTS_DIRECTORY,
    database_path=DATABASE_PATH,
):
    source_files = list_source_files(source_directory)
    current_chunks = current_processed_chunk_identity(source_directory)
    indexed_chunks = indexed_chunk_identity(database_path)
    current_by_source = {}
    indexed_by_source = {}
    for source, chunk_number, text in current_chunks:
        current_by_source.setdefault(source, []).append((chunk_number, text))
    for source, chunk_number, text in indexed_chunks:
        indexed_by_source.setdefault(source, []).append((chunk_number, text))

    source_states = {
        source: (
            "Indexed"
            if indexed_by_source.get(source) == current_by_source[source]
            else "Pending update"
            if source in indexed_by_source
            else "Pending addition"
        )
        for source in source_files
    }

    return {
        "source_files": source_files,
        "source_document_count": len(source_files),
        "source_chunk_count": len(current_chunks),
        "indexed_document_count": len({row[0] for row in indexed_chunks}),
        "indexed_chunk_count": len(indexed_chunks),
        "rebuild_required": current_chunks != indexed_chunks,
        "source_states": source_states,
        "pending_removed_sources": sorted(
            set(indexed_by_source) - set(current_by_source)
        ),
    }
