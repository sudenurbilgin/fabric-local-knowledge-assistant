# Fabric Local Knowledge Assistant

A private, local Retrieval-Augmented Generation (RAG) assistant for asking questions about your own documents with inspectable citations.

The application uses Microsoft Foundry Local for local retrieval and answer generation. Documents, embeddings, retrieval, and generated answers remain on the local machine; no cloud LLM API is used for question answering.

## Features

- Local document Q&A with grounded answers
- Numbered citations linked to retrieved evidence
- Refusal of questions that are not supported by the indexed documents
- Persistent semantic retrieval for faster warm queries
- Safe document management from the Streamlit interface
- Explicit, transactional knowledge-base rebuilds
- Dynamic document collections
- Markdown, TXT, text-based PDF, and DOCX support
- Local SQLite knowledge-base storage

## Supported documents

| Format | Support |
| --- | --- |
| Markdown (`.md`) | Yes |
| Plain text (`.txt`) | Yes |
| Text-based PDF (`.pdf`) | Yes |
| Word (`.docx`) | Yes |

Scanned or image-only PDFs are not currently supported because OCR is not implemented.

## Current production configuration

| Component | Configuration |
| --- | --- |
| Embedding model | `qwen3-embedding-0.6b` |
| Chat model | `Phi-4-mini-instruct-generic-cpu:5` |
| Execution provider | `CPUExecutionProvider` |
| Retrieval | Top-3 cosine similarity |
| Knowledge base | SQLite (`rag.db`) |
| Interface | Streamlit |
| Cloud LLM API | None |

The bundled example corpus currently contains 8 Microsoft Fabric documents and 90 indexed passages. These counts are discovered dynamically and are not hard-coded production requirements.

## Quick start

### Requirements

- Windows
- Python 3.12
- Microsoft Foundry Local

Clone the repository:

```powershell
git clone https://github.com/sudenurbilgin/fabric-local-knowledge-assistant.git
cd fabric-local-knowledge-assistant
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies and the local package:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

The configured embedding and chat model variants must be available through Microsoft Foundry Local.

### Run the Streamlit application

```powershell
.\run_app.ps1
```

Or launch Streamlit directly:

```powershell
python -m streamlit run streamlit_app.py
```

Then open the local Streamlit address shown in the terminal.

### Command-line interface

```powershell
python cli.py
```

## Knowledge-base workflow

Documents can be added or removed from the **Knowledge Base** page.

Changes to the source collection do not silently modify the active index. When the source collection differs from the current SQLite knowledge base, the application shows **Rebuild required**.

A rebuild extracts the current documents, cleans and chunks their text, generates local embeddings, validates the new data, and transactionally replaces the existing index.

If rebuilding fails, the previous valid knowledge base is preserved.

## Performance

The production retriever keeps the embedding model and stored vectors available between queries instead of recreating them for every request.

Historical warm-query benchmark on the original corpus:

| Retrieval implementation | Average |
| --- | ---: |
| Previous stateless retrieval | ~2.31 s |
| Persistent retrieval | ~0.38 s |
| Measured speedup | **6.06×** |

These are historical engineering measurements, not guaranteed latency for every query or machine. Cold model initialization is excluded.

## Validation

The repository includes engineering diagnostics covering:

- retrieval regression
- persistent retriever lifecycle and cleanup
- Knowledge Base management safety
- multi-format source ingestion
- dynamic corpus handling
- Streamlit presentation and session lifecycle

Experimental adaptive retrieval and neighboring-chunk expansion remain isolated from the production path.

## Known limitations

- Production retrieval intentionally uses a fixed Top-3 context, so broad questions whose required evidence spans multiple chunk boundaries may miss part of the available evidence.
- OCR for scanned or image-only PDFs is not implemented.
- Complex Word layouts, embedded images, comments, and tracked-change semantics are not reconstructed.

## Project structure

```text
app_pages/          Streamlit application pages
data/documents/     Local source documents
src/fabric_rag/     Production RAG package
diagnostics/        Engineering and regression diagnostics
experiments/        Historical experiments
rag.db              Current SQLite knowledge base
streamlit_app.py    Streamlit entry point
cli.py              Command-line interface
run_app.ps1         Windows launcher
```

## Disclaimer

This is an independent project and is not an official Microsoft product.

Bundled Microsoft Fabric documentation remains subject to its original licensing and attribution requirements.
