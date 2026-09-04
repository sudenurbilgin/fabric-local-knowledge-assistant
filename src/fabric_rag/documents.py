import re
import sys

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENTS_DIRECTORY,
)
from .ingestion import discover_source_paths, load_source_document



def parse_markdown_heading(line):
    match = re.match(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", line.rstrip("\n"))
    if match is None:
        return None

    title = re.sub(r"[ \t]+#+[ \t]*$", "", match.group(2)).strip()
    return len(match.group(1)), title


def fence_character(line):
    match = re.match(r"^[ \t]*(`{3,}|~{3,})", line)
    return match.group(1)[0] if match else None


def remove_related_content_sections(text):
    lines = text.splitlines(keepends=True)
    cleaned_lines = []
    index = 0
    active_fence = None

    while index < len(lines):
        marker = fence_character(lines[index])
        if marker is not None:
            active_fence = None if active_fence == marker else marker
            cleaned_lines.append(lines[index])
            index += 1
            continue

        heading = parse_markdown_heading(lines[index]) if active_fence is None else None
        if heading is None or heading[1].casefold() != "related content":
            cleaned_lines.append(lines[index])
            index += 1
            continue

        section_level = heading[0]
        index += 1
        skipped_fence = None

        while index < len(lines):
            marker = fence_character(lines[index])
            if marker is not None:
                skipped_fence = None if skipped_fence == marker else marker
                index += 1
                continue

            next_heading = (
                parse_markdown_heading(lines[index])
                if skipped_fence is None
                else None
            )
            if next_heading is not None and next_heading[0] <= section_level:
                break

            index += 1

    return "".join(cleaned_lines)


def load_documents(directory):
    document_paths = discover_source_paths(directory)
    if not document_paths:
        raise RuntimeError(
            f"No supported source documents were found in {directory}."
        )

    return [load_source_document(path) for path in document_paths]


def clean_markdown(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.removeprefix("\ufeff")

    text = re.sub(
        r"\A---[ \t]*\n.*?\n---[ \t]*(?:\n|$)",
        "",
        text,
        count=1,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"(?ms)^[ \t]*:::image\b.*?:::[ \t]*(?:\n|$)",
        "",
        text,
    )
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(
        r"(?m)^>[ \t]*\[!NOTE\][ \t]*\n"
        r">[ \t]*\*\*Share your feedback and shape the future of Fabric\*\*[ \t]*\n"
        r"(?:^>.*(?:\n|$))*",
        "",
        text,
    )
    text = remove_related_content_sections(text)
    text = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", text)

    return text.strip()


def split_oversized_paragraph(paragraph, max_chars):
    parts = []
    remaining = paragraph.strip()

    while len(remaining) > max_chars:
        window = remaining[: max_chars + 1]
        split_at = max(window.rfind("\n"), window.rfind(" "))
        if split_at < max_chars // 2:
            split_at = max_chars

        part = remaining[:split_at].strip()
        if part:
            parts.append(part)
        remaining = remaining[split_at:].strip()

    if remaining:
        parts.append(remaining)

    return parts


def overlap_from_chunk(chunk, overlap_chars):
    if overlap_chars == 0:
        return ""

    start = max(0, len(chunk) - overlap_chars)
    paragraph_boundary = chunk.find("\n\n", start)
    if paragraph_boundary != -1:
        start = paragraph_boundary + 2

    return chunk[start:].strip()


def chunk_text(text, max_chars=CHUNK_SIZE, overlap_chars=CHUNK_OVERLAP):
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero.")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be nonnegative and smaller than max_chars.")

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]
    units = [
        part
        for paragraph in paragraphs
        for part in split_oversized_paragraph(paragraph, max_chars)
    ]

    chunks = []
    current = ""

    for unit in units:
        candidate = f"{current}\n\n{unit}" if current else unit
        if current and len(candidate) > max_chars:
            chunks.append(current)

            overlap = overlap_from_chunk(current, overlap_chars)
            available_overlap = max_chars - len(unit) - 2
            if available_overlap <= 0:
                overlap = ""
            elif len(overlap) > available_overlap:
                overlap = overlap[-available_overlap:].lstrip()

            current = f"{overlap}\n\n{unit}" if overlap else unit
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks


def process_documents(documents):
    if not documents:
        raise RuntimeError("At least one source document is required.")

    processed = []

    for document in documents:
        cleaned_text = clean_markdown(document["raw_text"])
        chunk_contents = chunk_text(cleaned_text)
        if not chunk_contents:
            raise RuntimeError(
                f"Document '{document['source']}' produced no non-empty chunks."
            )
        chunks = [
            {
                "source": document["source"],
                "chunk_number": index,
                "text": chunk,
            }
            for index, chunk in enumerate(chunk_contents, start=1)
        ]
        processed.append(
            {
                **document,
                "cleaned_text": cleaned_text,
                "chunks": chunks,
            }
        )

    validate_processed_documents(processed)
    return processed


def validate_processed_documents(processed_documents):
    chunks = [
        chunk
        for document in processed_documents
        for chunk in document["chunks"]
    ]
    if not chunks:
        raise RuntimeError("The document collection produced no chunks.")

    seen_keys = set()
    for chunk in chunks:
        source = chunk.get("source")
        chunk_number = chunk.get("chunk_number")
        text = chunk.get("text")

        if not isinstance(source, str) or not source.strip():
            raise RuntimeError("Every chunk must have a non-empty source filename.")
        if not isinstance(chunk_number, int) or chunk_number <= 0:
            raise RuntimeError("Every chunk must have a positive integer chunk number.")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Every chunk must contain non-empty text.")
        if len(text) > CHUNK_SIZE:
            raise RuntimeError(
                f"Chunk {source} #{chunk_number} exceeds {CHUNK_SIZE} characters."
            )

        key = (source, chunk_number)
        if key in seen_keys:
            raise RuntimeError(
                f"Duplicate chunk key found for {source}, chunk {chunk_number}."
            )
        seen_keys.add(key)


def print_example(chunk):
    print("=" * 72)
    print(f"SOURCE: {chunk['source']}")
    print(f"CHUNK: {chunk['chunk_number']}")
    print(f"CHARACTERS: {len(chunk['text'])}")
    print()
    print(chunk["text"])
    print()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    documents = load_documents(DOCUMENTS_DIRECTORY)
    processed_documents = process_documents(documents)

    print(f"Documents loaded: {len(processed_documents)}\n")

    for document in processed_documents:
        print(f"Filename: {document['source']}")
        print(f"Raw characters: {len(document['raw_text'])}")
        print(f"Cleaned characters: {len(document['cleaned_text'])}")
        print(f"Chunks: {len(document['chunks'])}\n")

    total_chunks = sum(len(document["chunks"]) for document in processed_documents)
    print(f"Total chunks: {total_chunks}\n")

    documents_by_source = {
        document["source"]: document for document in processed_documents
    }
    fabric_chunks = documents_by_source["microsoft-fabric-overview.md"]["chunks"]
    data_factory_chunks = documents_by_source["data-factory-overview.md"]["chunks"]
    direct_lake_chunks = documents_by_source["direct-lake-overview.md"]["chunks"]

    examples = [
        fabric_chunks[0],
        data_factory_chunks[len(data_factory_chunks) // 2],
        direct_lake_chunks[0],
    ]

    print("Example chunks:\n")
    for example in examples:
        print_example(example)


if __name__ == "__main__":
    main()
