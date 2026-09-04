import sys
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for import_path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from app_pages import shared
from fabric_rag.rag import RAGSession
from fabric_rag.retrieval import PersistentRetriever


class FakeModel:
    def __init__(self, owner, client_attribute, events):
        self.owner = owner
        self.client_attribute = client_attribute
        self.events = events
        self.is_loaded = True
        self.unload_count = 0

    def unload(self):
        if getattr(self.owner, self.client_attribute) is not None:
            raise AssertionError("The client reference was not released before unload.")
        self.events.append("model_unload")
        self.unload_count += 1
        self.is_loaded = False


class FakeRetriever:
    def __init__(self, events):
        self.events = events
        self.close_count = 0

    def close(self):
        self.events.append("retriever_close")
        self.close_count += 1


class FakeSession:
    def __init__(self):
        self.close_count = 0

    def close(self):
        self.close_count += 1


def validate_persistent_retriever_close():
    events = []
    retriever = PersistentRetriever.__new__(PersistentRetriever)
    retriever._stored_chunks = [object()]
    retriever._embedding_client = object()
    retriever._embedding_model = FakeModel(
        retriever,
        "_embedding_client",
        events,
    )
    embedding_model = retriever._embedding_model
    retriever._closed = False

    retriever.close()
    retriever.close()

    return (
        events == ["model_unload"]
        and embedding_model.unload_count == 1
        and retriever._embedding_client is None
        and retriever._embedding_model is None
        and retriever._stored_chunks is None
        and retriever._closed
    )


def validate_rag_session_close():
    events = []
    session = RAGSession.__new__(RAGSession)
    session._retriever = FakeRetriever(events)
    retriever = session._retriever
    session._chat_client = object()
    session._chat_model = FakeModel(session, "_chat_client", events)
    chat_model = session._chat_model
    session._closed = False

    session.close()
    session.close()

    return (
        events == ["retriever_close", "model_unload"]
        and retriever.close_count == 1
        and chat_model.unload_count == 1
        and session._retriever is None
        and session._chat_client is None
        and session._closed
    )


def validate_streamlit_session_cleanup():
    session = FakeSession()
    messages = [{"role": "user", "content": "Keep this history."}]
    state = {"rag_session": session, "messages": messages}

    with patch.object(shared.st, "session_state", state):
        first_result = shared.close_active_rag_session()
        second_result = shared.close_active_rag_session()

    return (
        first_result is True
        and second_result is False
        and session.close_count == 1
        and "rag_session" not in state
        and state["messages"] == messages
    )


def main():
    results = {
        "Persistent retriever releases client before one model unload": (
            validate_persistent_retriever_close()
        ),
        "RAG session closes retriever then releases chat client before unload": (
            validate_rag_session_close()
        ),
        "Streamlit reset removes only the session and preserves chat history": (
            validate_streamlit_session_cleanup()
        ),
    }

    print("SESSION RESET MODEL-FREE VALIDATION")
    print("=" * 50)
    for name, passed in results.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    passed_count = sum(results.values())
    print(f"\nPassed: {passed_count} / {len(results)}")
    if passed_count != len(results):
        raise RuntimeError("At least one session-reset lifecycle check failed.")


if __name__ == "__main__":
    main()
