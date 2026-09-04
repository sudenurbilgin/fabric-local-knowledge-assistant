import logging
import re
import sqlite3
from pathlib import Path

import streamlit as st

from fabric_rag.config import RETRIEVAL_TOP_K
from fabric_rag.knowledge_base import get_knowledge_base_statistics
from fabric_rag.management import get_knowledge_base_management_status


LOGGER = logging.getLogger(__name__)
SOURCE_LABEL_PATTERN = re.compile(rf"\[([1-{RETRIEVAL_TOP_K}])\]")
FILE_TYPE_LABELS = {
    ".md": "Markdown",
    ".txt": "Plain text",
    ".pdf": "PDF",
    ".docx": "Word document",
}
SPECIAL_TITLE_WORDS = {
    "api": "API",
    "docx": "DOCX",
    "fabric": "Fabric",
    "lakehouse": "Lakehouse",
    "microsoft": "Microsoft",
    "onelake": "OneLake",
    "pdf": "PDF",
    "sql": "SQL",
}
FRIENDLY_TITLE_OVERRIDES = {
    "direct-lake-how-it-works": "Direct Lake: How It Works",
}


def initialize_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []


def read_knowledge_base_metadata():
    try:
        statistics = get_knowledge_base_statistics()
        return statistics["document_count"], statistics["chunk_count"]
    except (OSError, sqlite3.Error):
        LOGGER.warning("Could not read knowledge-base metadata.", exc_info=True)
        return None, None


def read_management_status():
    try:
        return get_knowledge_base_management_status()
    except Exception:
        LOGGER.warning(
            "Could not compare source documents with the local index.",
            exc_info=True,
        )
        return None


def indexed_counts(management_status):
    if management_status is not None:
        return (
            management_status["indexed_document_count"],
            management_status["indexed_chunk_count"],
        )
    return read_knowledge_base_metadata()


def close_active_rag_session():
    session = st.session_state.get("rag_session")
    if session is None:
        return False
    try:
        session.close()
    finally:
        st.session_state.pop("rag_session", None)
    return True


def source_labels(answer):
    return {int(label) for label in SOURCE_LABEL_PATTERN.findall(answer or "")}


def split_retrieved_chunks(answer, chunks):
    labels = source_labels(answer)
    cited = []
    uncited = []
    for rank, chunk in enumerate(chunks, start=1):
        (cited if rank in labels else uncited).append((rank, chunk))
    return cited, uncited


def friendly_source_title(filename):
    stem = Path(filename).stem
    if stem.casefold() in FRIENDLY_TITLE_OVERRIDES:
        return FRIENDLY_TITLE_OVERRIDES[stem.casefold()]
    words = re.split(r"[-_\s]+", stem)
    return " ".join(
        SPECIAL_TITLE_WORDS.get(word.casefold(), word.capitalize())
        for word in words
        if word
    )


def source_type_label(filename):
    return FILE_TYPE_LABELS.get(Path(filename).suffix.casefold(), "Document")


def text_excerpt(text, max_characters=320):
    compact = " ".join((text or "").split())
    if len(compact) <= max_characters:
        return compact
    return compact[: max_characters - 3].rstrip() + "..."


def escape_markdown(text):
    return re.sub(r"([\\`*_{}\[\]()#+.!|>-])", r"\\\1", text)


def format_file_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def set_notice(key, level, message):
    st.session_state[key] = {"level": level, "message": message}


def render_notice(key):
    notice = st.session_state.pop(key, None)
    if notice is not None:
        getattr(st, notice["level"])(notice["message"])


def apply_visual_polish():
    st.markdown(
        """
        <style>
        :root {
            --aurora-coral: #F07E67;
            --aurora-coral-soft: rgba(240, 126, 103, 0.14);
            --aurora-mint: #79CEB2;
            --aurora-mint-soft: rgba(121, 206, 178, 0.12);
            --aurora-lilac: #B59AE6;
            --aurora-lilac-soft: rgba(181, 154, 230, 0.12);
            --aurora-ink: #10121A;
            --aurora-surface: #191A24;
            --aurora-surface-raised: #1E1E29;
            --aurora-text: #F4EEE8;
            --aurora-muted: #B8B1B5;
            --aurora-border: #413640;
            --aurora-success: #70C99A;
            --aurora-warning: #E4AD62;
        }
        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 16% 0%, rgba(240, 126, 103, 0.07), transparent 30rem),
                var(--aurora-ink);
        }
        [data-testid="stHeader"] {
            background: rgba(16, 18, 26, 0.96);
        }
        .block-container {
            max-width: 1120px;
            padding-top: 1.15rem;
            padding-bottom: 7rem;
        }
        h1 {
            font-size: 2.05rem;
            letter-spacing: -0.025em;
            margin-bottom: 0.35rem;
        }
        h2 {
            margin-top: 1.35rem;
        }
        [data-testid="stChatMessage"] {
            padding-bottom: 0.35rem;
            padding-top: 0.35rem;
        }
        [data-testid="stChatMessageAvatarUser"] {
            background: rgba(240, 126, 103, 0.20);
            color: var(--aurora-coral);
        }
        [data-testid="stChatMessageAvatarAssistant"] {
            background: rgba(121, 206, 178, 0.17);
            color: var(--aurora-mint);
        }
        [data-testid="stMetric"] {
            background: linear-gradient(145deg, rgba(240, 126, 103, 0.055), rgba(181, 154, 230, 0.035));
            border-color: var(--aurora-border);
            border-radius: 0.9rem;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.55rem;
            color: var(--aurora-text);
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(30, 30, 41, 0.72);
            border-color: var(--aurora-border);
            border-radius: 0.9rem;
        }
        .stButton > button p {
            line-height: 1.25;
            white-space: normal;
        }
        .stButton > button {
            border-radius: 0.7rem;
        }
        blockquote {
            border-left-color: var(--aurora-lilac) !important;
            color: #D7D0D3 !important;
            line-height: 1.55;
        }
        .chat-input-clearance {
            height: 5.5rem;
        }
        .st-key-chat_controls {
            align-items: center;
            background: rgba(25, 26, 36, 0.72);
            border: 1px solid rgba(240, 126, 103, 0.18);
            border-radius: 0.8rem;
            margin-left: auto;
            margin-top: 0.45rem;
            padding: 0.18rem 0.35rem;
            width: fit-content;
        }
        .fabric-aurora-hero {
            align-items: center;
            background:
                radial-gradient(circle at 12% 10%, rgba(240, 126, 103, 0.16), transparent 36%),
                radial-gradient(circle at 88% 20%, rgba(181, 154, 230, 0.11), transparent 34%),
                rgba(25, 26, 36, 0.66);
            border: 1px solid rgba(240, 126, 103, 0.22);
            border-radius: 1.15rem;
            display: flex;
            gap: 1rem;
            margin: 0.1rem 0 0.85rem;
            min-height: 5.8rem;
            padding: 0.75rem 10rem 0.75rem 1.1rem;
            position: relative;
        }
        .aurora-mark {
            flex: 0 0 3.35rem;
            height: 3.35rem;
            position: relative;
            width: 3.35rem;
        }
        .aurora-mark::before,
        .aurora-mark::after {
            background: linear-gradient(90deg, var(--aurora-coral), var(--aurora-lilac));
            content: "";
            height: 2px;
            left: 0.7rem;
            opacity: 0.65;
            position: absolute;
            top: 1.65rem;
            transform: rotate(32deg);
            width: 2rem;
        }
        .aurora-mark::after {
            background: linear-gradient(90deg, var(--aurora-mint), var(--aurora-lilac));
            transform: rotate(-32deg);
        }
        .aurora-node {
            border-radius: 0.45rem;
            box-shadow: 0 0 0 4px rgba(16, 18, 26, 0.72);
            height: 1.05rem;
            position: absolute;
            width: 1.05rem;
        }
        .aurora-node-coral {
            background: var(--aurora-coral);
            left: 0.05rem;
            top: 0.25rem;
        }
        .aurora-node-mint {
            background: var(--aurora-mint);
            right: 0.05rem;
            top: 0.25rem;
        }
        .aurora-node-lilac {
            background: var(--aurora-lilac);
            bottom: 0.15rem;
            left: 1.15rem;
        }
        .aurora-hero-copy {
            min-width: 0;
        }
        .aurora-hero-copy h1 {
            color: var(--aurora-text);
            font-size: 1.85rem;
            line-height: 1.12;
            margin: 0 0 0.3rem;
        }
        .aurora-hero-copy p {
            color: var(--aurora-muted);
            line-height: 1.45;
            margin: 0;
        }
        .aurora-hero-copy .hero-primary {
            color: var(--aurora-text);
        }
        .aurora-badge {
            border: 1px solid rgba(121, 206, 178, 0.35);
            border-radius: 999px;
            color: var(--aurora-mint);
            display: inline-block;
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.11em;
            margin-top: 0;
            padding: 0.2rem 0.48rem;
            position: absolute;
            right: 1.1rem;
            top: 50%;
            transform: translateY(-50%);
        }
        .evidence-heading {
            align-items: center;
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-bottom: 0.35rem;
        }
        .citation-badge {
            background: var(--aurora-coral-soft);
            border: 1px solid rgba(240, 126, 103, 0.38);
            border-radius: 999px;
            color: var(--aurora-coral);
            font-size: 0.78rem;
            font-weight: 750;
            padding: 0.16rem 0.44rem;
        }
        .evidence-title {
            color: var(--aurora-text);
            font-weight: 650;
        }
        .source-pill {
            background: var(--aurora-lilac-soft);
            border-radius: 999px;
            color: #CDBDF0;
            font-size: 0.72rem;
            padding: 0.14rem 0.42rem;
        }
        .architecture-flow {
            display: flex;
            align-items: center;
            gap: 0.45rem;
            margin: 0.35rem 0 0.7rem;
            width: 100%;
        }
        .architecture-stage {
            align-items: center;
            background: var(--aurora-coral-soft);
            border: 1px solid rgba(240, 126, 103, 0.28);
            border-radius: 0.65rem;
            display: flex;
            flex: 1 1 0;
            justify-content: center;
            min-height: 3.5rem;
            min-width: 0;
            padding: 0.55rem 0.4rem;
            text-align: center;
        }
        .architecture-stage span {
            font-size: 0.82rem;
            font-weight: 600;
            line-height: 1.2;
        }
        .architecture-stage.tone-mint {
            background: var(--aurora-mint-soft);
            border-color: rgba(121, 206, 178, 0.28);
        }
        .architecture-stage.tone-lilac {
            background: var(--aurora-lilac-soft);
            border-color: rgba(181, 154, 230, 0.28);
        }
        .architecture-arrow {
            color: rgba(216, 199, 206, 0.72);
            flex: 0 0 auto;
            font-size: 1.05rem;
        }
        @media (max-width: 800px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
            .architecture-flow {
                align-items: stretch;
                flex-direction: column;
                gap: 0.25rem;
            }
            .architecture-stage {
                flex-basis: auto;
                min-height: 2.7rem;
                width: 100%;
            }
            .architecture-arrow {
                align-self: center;
                transform: rotate(90deg);
            }
            .fabric-aurora-hero {
                align-items: flex-start;
                padding-right: 1rem;
            }
            .aurora-hero-copy h1 {
                font-size: 1.65rem;
            }
            .aurora-badge {
                margin-top: 0.45rem;
                position: static;
                transform: none;
            }
            
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
