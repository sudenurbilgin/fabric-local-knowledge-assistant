# Fabric Local Knowledge Assistant

A private, local retrieval-augmented generation assistant that answers questions from your documents with inspectable citations using Microsoft Foundry Local.

## Highlights

- Local answer generation with `Phi-4-mini-instruct-generic-cpu:5`
- Local semantic retrieval with `qwen3-embedding-0.6b`
- Grounded numbered citations and unsupported-question refusal
- Persistent retriever with measured warm-query reuse
- Transactional SQLite knowledge-base rebuilds
- Safe, staged source management from Streamlit
- Markdown, plain text, text-based PDF, and DOCX ingestion
- No cloud LLM API

## Product interface

The Streamlit application uses three focused pages:

- **Chat** — cited answers, supporting evidence cards, examples, and session controls
- **Knowledge Base** — source management, index status, coverage charts, and explicit rebuilds
- **System & Evaluation** — architecture, current configuration, historical benchmark, and validation record

### Demo screenshots

> Presentation placeholder: add final Chat, Knowledge Base, and System & Evaluation screenshots here before publishing the project portfolio.

## Architecture

```text
Documents
    → text extraction and Markdown cleaning
    → paragraph-aware chunking
    → qwen3 embeddings
    → SQLite knowledge base
    → Top-3 cosine-similarity retrieval
    → local Phi-4 Mini generation
    → cited answer
```

The bundled corpus currently contains eight selected Microsoft Fabric documents and 90 indexed passages. Those counts are discovered dynamically and are not production requirements.

## Local and private by design

Source extraction, chunking, embeddings, retrieval, and answer generation run through the local project pipeline. Uploaded source files remain in `data/documents`, and the generated vectors remain in `rag.db`. The application does not send documents or questions to a cloud LLM API. Initial model catalog access or acquisition can still require the Microsoft Foundry Local runtime to reach its configured catalog.

## Supported documents

| Format | Behavior |
| --- | --- |
| Markdown (`.md`) | Common UTF-8, UTF-16, and Windows-1254 encodings are handled where possible. |
| Plain text (`.txt`) | Uses the same deterministic encoding strategy as Markdown. |
| Text-based PDF (`.pdf`) | Extracts embedded text page by page and ignores empty pages. |
| Word (`.docx`) | Preserves paragraph order and standard heading/list text. |

Scanned or image-only PDF OCR is not supported. Complex Word layout, images, comments, and tracked-change semantics are not reconstructed.

## Historical engineering benchmark

Measured on the original baseline corpus with warm-query retrieval only:

| Retrieval implementation | Average time |
| --- | ---: |
| Previous stateless retrieval | approximately 2.31 seconds |
| Persistent retrieval | approximately 0.38 seconds |
| Measured warm speedup | **6.06x** |

These are historical engineering measurements, not a guarantee for every query. Cold model initialization is excluded.

## Engineering validation

| Diagnostic | Recorded result |
| --- | ---: |
| Top-3 persistent retrieval regression | 4 / 4 PASS |
| Knowledge Base Manager safety | 12 / 12 PASS |
| Multi-format source ingestion | 13 / 13 PASS |
| Dynamic corpus validation | PASS |
| Persistent retriever reuse and cleanup | PASS |

These are engineering regression results, not accuracy, confidence, or answer-quality percentages. Experimental adaptive and neighboring-chunk retrieval remain isolated in diagnostics.

## Installation

This project targets Windows with Python 3.12 and Microsoft Foundry Local.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
```

The required embedding and chat model variants must be available through Microsoft Foundry Local.

## Run

Presentation-friendly PowerShell launcher:

```powershell
.\run_app.ps1
```

Or launch Streamlit directly:

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Command-line interface:

```powershell
.\.venv\Scripts\python.exe cli.py
```

## Knowledge-base workflow

Source additions and removals are staged and never silently treated as indexed. The application compares the current extracted/chunked content with `rag.db` and shows **Rebuild required** after an addition, removal, or content modification.

An explicit rebuild closes the active RAG session, extracts the current sources, generates embeddings, and transactionally validates the replacement index. If rebuilding fails, the previous valid database is preserved. The next question after a successful rebuild creates a fresh local session lazily.

## Known limitation

Broad questions whose required evidence spans chunk boundaries may not receive every required passage within the fixed Top-3 context.

## Repository structure

```text
app_pages/                         Streamlit Chat, Knowledge Base, and System pages
data/documents/                    Local source-of-truth document collection
src/fabric_rag/config.py           Paths and validated production constants
src/fabric_rag/ingestion.py        Multi-format extraction and text decoding
src/fabric_rag/documents.py        Cleaning and paragraph-aware chunking
src/fabric_rag/embeddings.py       Embedding preparation and model helpers
src/fabric_rag/knowledge_base.py   Transactional SQLite build and validation
src/fabric_rag/management.py       Safe source management and freshness checks
src/fabric_rag/retrieval.py        Persistent Top-3 cosine retrieval
src/fabric_rag/rag.py              Grounded prompt and persistent RAG session
diagnostics/                       Manual engineering and regression diagnostics
experiments/                       Historical learning exercises
streamlit_app.py                   Multipage Streamlit router
cli.py                             Persistent command-line interface
run_app.ps1                        Windows Streamlit launcher
rag.db                             Generated local knowledge base
```
