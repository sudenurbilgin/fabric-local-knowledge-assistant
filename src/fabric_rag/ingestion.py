import re
from io import BytesIO
from pathlib import Path

from docx import Document
from pypdf import PdfReader


SUPPORTED_SOURCE_SUFFIXES = frozenset({".md", ".txt", ".pdf", ".docx"})
SUPPORTED_UPLOAD_TYPES = ("md", "txt", "pdf", "docx")
OBVIOUS_BINARY_SIGNATURES = (
    b"%PDF-",
    b"PK\x03\x04",
    b"\x89PNG\r\n\x1a\n",
    b"\xff\xd8\xff",
    b"GIF87a",
    b"GIF89a",
)


def source_suffix(filename):
    suffix = Path(filename).suffix.casefold()
    if suffix not in SUPPORTED_SOURCE_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SOURCE_SUFFIXES))
        raise ValueError(f"Unsupported source type. Supported types: {supported}.")
    return suffix


def normalize_extracted_text(text):
    if not isinstance(text, str):
        raise TypeError("Extracted source content must be text.")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.removeprefix("\ufeff").strip()
    if not normalized:
        raise ValueError("No readable text was found in this source document.")
    return normalized


def usable_decoded_text(text):
    if not text.strip() or "\x00" in text:
        return False
    return not any(
        ord(character) < 32 and character not in "\t\n\r"
        or 127 <= ord(character) <= 159
        for character in text
    )


def inferred_utf16_encoding(content):
    sample = content[:4096]
    pair_count = len(sample) // 2
    if pair_count < 2:
        return None

    even_nulls = sample[0::2].count(0)
    odd_nulls = sample[1::2].count(0)
    minimum_nulls = max(2, pair_count // 4)
    if odd_nulls >= minimum_nulls and odd_nulls > even_nulls * 2:
        return "utf-16-le"
    if even_nulls >= minimum_nulls and even_nulls > odd_nulls * 2:
        return "utf-16-be"
    return None


def decode_text_bytes(content):
    if not isinstance(content, bytes):
        raise TypeError("Uploaded source content must be provided as bytes.")
    if not content:
        raise ValueError("The source document is empty.")
    if content.startswith(OBVIOUS_BINARY_SIGNATURES):
        raise ValueError("The text source appears to contain binary data.")

    if content.startswith(b"\xef\xbb\xbf"):
        candidates = ["utf-8-sig"]
    elif content.startswith((b"\xff\xfe", b"\xfe\xff")):
        candidates = ["utf-16"]
    else:
        inferred_utf16 = inferred_utf16_encoding(content)
        candidates = [inferred_utf16] if inferred_utf16 else []
        candidates.extend(["utf-8", "cp1254"])

    for encoding in candidates:
        try:
            decoded = content.decode(encoding, errors="strict")
        except (UnicodeDecodeError, UnicodeError):
            continue
        if usable_decoded_text(decoded):
            return normalize_extracted_text(decoded)

    raise ValueError(
        "The text file could not be decoded as UTF-8, UTF-16, or Windows-1254, "
        "or it appears to contain binary data."
    )


def extract_pdf_text(content):
    if not isinstance(content, bytes) or not content.startswith(b"%PDF-"):
        raise ValueError("The PDF is malformed or could not be read.")
    try:
        reader = PdfReader(BytesIO(content), strict=False)
    except Exception as error:
        raise ValueError("The PDF is malformed or could not be read.") from error

    if reader.is_encrypted:
        try:
            decrypted = reader.decrypt("")
        except Exception as error:
            raise ValueError("Password-protected PDFs are not supported.") from error
        if decrypted == 0:
            raise ValueError("Password-protected PDFs are not supported.")

    try:
        page_texts = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                page_texts.append(text.strip())
    except Exception as error:
        raise ValueError("The PDF is malformed or could not be read.") from error

    if not page_texts:
        raise ValueError(
            "No extractable text was found in this PDF. OCR is not supported yet."
        )
    return normalize_extracted_text("\n\n".join(page_texts))


def paragraph_prefix(paragraph):
    style_name = paragraph.style.name if paragraph.style is not None else ""
    heading_match = re.fullmatch(r"Heading ([1-6])", style_name, flags=re.IGNORECASE)
    if heading_match:
        return "#" * int(heading_match.group(1)) + " "

    paragraph_properties = paragraph._p.pPr
    has_numbering = (
        paragraph_properties is not None
        and paragraph_properties.numPr is not None
    )
    if has_numbering or style_name.casefold().startswith("list"):
        return "- "
    return ""


def extract_docx_text(content):
    try:
        document = Document(BytesIO(content))
        paragraphs = []
        for paragraph in document.paragraphs:
            text = paragraph.text.strip()
            if text:
                paragraphs.append(paragraph_prefix(paragraph) + text)
    except Exception as error:
        raise ValueError("The Word document is malformed or could not be read.") from error

    if not paragraphs:
        raise ValueError("No readable paragraph text was found in this Word document.")
    return normalize_extracted_text("\n\n".join(paragraphs))


def extract_source_text(filename, content):
    suffix = source_suffix(filename)
    if suffix in {".md", ".txt"}:
        return decode_text_bytes(content)
    if suffix == ".pdf":
        return extract_pdf_text(content)
    return extract_docx_text(content)


def discover_source_paths(source_directory):
    if not source_directory.exists():
        return []
    return sorted(
        (
            path
            for path in source_directory.iterdir()
            if path.is_file() and path.suffix.casefold() in SUPPORTED_SOURCE_SUFFIXES
        ),
        key=lambda path: (path.name.casefold(), path.name),
    )


def load_source_document(path):
    return {
        "source": path.name,
        "raw_text": extract_source_text(path.name, path.read_bytes()),
    }
