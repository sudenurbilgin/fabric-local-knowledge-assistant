import logging
import time
from html import escape

import streamlit as st

from fabric_rag.rag import RAGSession

from .shared import (
    close_active_rag_session,
    escape_markdown,
    friendly_source_title,
    indexed_counts,
    read_management_status,
    source_type_label,
    split_retrieved_chunks,
    text_excerpt,
)


LOGGER = logging.getLogger(__name__)
EXAMPLE_QUESTIONS = (
    "What is OneLake?",
    "How does Direct Lake work?",
    "What is Data Factory in Microsoft Fabric?",
    "Compare Lakehouse and Data Warehouse.",
)
ANSWER_PROGRESS = {
    "loading_retrieval_model": "Loading the local retrieval model...",
    "retrieving_evidence": "Retrieving evidence...",
    "generating_grounded_answer": "Generating a grounded answer...",
}
CHAT_AVATARS = {
    "user": ":material/person:",
    "assistant": ":material/hub:",
}


def render_evidence_card(rank, chunk):
    title = friendly_source_title(chunk["source"])
    excerpt = escape_markdown(text_excerpt(chunk["text"]))
    with st.container(border=True):
        st.markdown(
            '<div class="evidence-heading">'
            f'<span class="citation-badge">[{rank}]</span>'
            f'<span class="evidence-title">{escape(title)}</span>'
            f'<span class="source-pill">{escape(source_type_label(chunk["source"]))}</span>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"> {excerpt}")
        with st.expander("View evidence"):
            st.text(chunk["text"])
        with st.expander("Technical details"):
            st.text(f"Original filename: {chunk['source']}")
            st.text(f"Retrieval rank: {rank}")
            st.text(f"Chunk number: {chunk['chunk_number']}")
            st.text(f"Cosine similarity: {chunk['similarity']:.6f}")


def render_sources(answer, chunks):
    cited, uncited = split_retrieved_chunks(answer, chunks)
    if cited:
        st.markdown("#### Supporting evidence")
        for rank, chunk in cited:
            render_evidence_card(rank, chunk)

    if uncited:
        title = (
            "Other retrieved context"
            if cited
            else "Retrieved context not used as supporting evidence"
        )
        with st.expander(title):
            st.caption(
                "These passages are diagnostic retrieval context and are not "
                "presented as support for the answer."
            )
            for position, (rank, chunk) in enumerate(uncited):
                st.markdown(
                    f"**[{rank}] "
                    f"{escape_markdown(friendly_source_title(chunk['source']))}**"
                )
                st.caption(
                    f"{source_type_label(chunk['source'])} · "
                    f"{chunk['source']} · chunk {chunk['chunk_number']} · "
                    f"cosine similarity {chunk['similarity']:.6f}"
                )
                st.text(text_excerpt(chunk["text"], max_characters=420))
                if position < len(uncited) - 1:
                    st.divider()


def render_message(message):
    with st.chat_message(
        message["role"],
        avatar=CHAT_AVATARS.get(message["role"]),
    ):
        if message.get("error"):
            st.error(message["content"])
            return

        st.markdown(message["content"])
        if message["role"] == "assistant":
            if message.get("response_time") is not None:
                st.caption(
                    f"Answered in {message['response_time']:.1f} s · Local inference"
                )
            if message.get("retrieved_chunks"):
                render_sources(message["content"], message["retrieved_chunks"])


def get_or_create_rag_session():
    session = st.session_state.get("rag_session")
    if session is not None:
        return session

    with st.status("Loading local models...", expanded=True) as status:
        status.write("Starting the local answer model...")
        try:
            session = RAGSession()
        except Exception:
            status.update(label="Local models could not be loaded", state="error")
            raise
        status.update(label="Local answer model ready", state="complete", expanded=False)

    st.session_state.rag_session = session
    return session


def handle_question(question):
    user_message = {"role": "user", "content": question}
    st.session_state.messages.append(user_message)
    render_message(user_message)

    with st.chat_message("assistant", avatar=CHAT_AVATARS["assistant"]):
        try:
            session = get_or_create_rag_session()
            with st.status("Preparing a grounded answer...", expanded=True) as status:
                started = time.perf_counter()
                result = session.answer(
                    question,
                    progress_callback=lambda stage: status.write(
                        ANSWER_PROGRESS[stage]
                    ),
                )
                response_time = time.perf_counter() - started
                status.update(label="Answer ready", state="complete", expanded=False)

            assistant_message = {
                "role": "assistant",
                "content": result["answer"],
                "retrieved_chunks": result["retrieved_chunks"],
                "model": result["model"],
                "execution_provider": result["execution_provider"],
                "response_time": response_time,
            }
            st.markdown(assistant_message["content"])
            st.caption(f"Answered in {response_time:.1f} s · Local inference")
            render_sources(
                assistant_message["content"],
                assistant_message["retrieved_chunks"],
            )
        except Exception:
            LOGGER.exception("The local RAG backend could not answer the question.")
            assistant_message = {
                "role": "assistant",
                "content": (
                    "The local assistant could not complete that request. "
                    "Please try again or reset the local session."
                ),
                "error": True,
            }
            st.error(assistant_message["content"])

    st.session_state.messages.append(assistant_message)


def render_health_metrics():
    management_status = read_management_status()
    document_count, passage_count = indexed_counts(management_status)
    session_ready = st.session_state.get("rag_session") is not None
    index_status = (
        "Unavailable"
        if management_status is None
        else "Rebuild required"
        if management_status["rebuild_required"]
        else "Up to date"
    )

    columns = st.columns(4)
    columns[0].metric(
        "Local inference",
        "Ready" if session_ready else "Ready on demand",
        border=True,
        icon=":material/offline_bolt:",
    )
    columns[1].metric(
        "Indexed sources",
        document_count if document_count is not None else "—",
        border=True,
        icon=":material/description:",
    )
    columns[2].metric(
        "Indexed passages",
        passage_count if passage_count is not None else "—",
        border=True,
        icon=":material/segment:",
    )
    columns[3].metric(
        "Knowledge base",
        index_status,
        border=True,
        icon=(
            ":material/warning:"
            if index_status == "Rebuild required"
            else ":material/check_circle:"
        ),
    )

    if management_status is not None and management_status["rebuild_required"]:
        st.warning(
            "Rebuild required. Chat continues to use the previous valid index "
            "until you rebuild it on the Knowledge Base page."
        )


def render_chat_controls():
    with st.container(
        key="chat_controls",
        horizontal=True,
        horizontal_alignment="right",
        vertical_alignment="center",
        gap="small",
    ):
        if st.button(
            "Clear",
            type="tertiary",
            icon=":material/delete_sweep:",
            key="clear_chat",
        ):
            st.session_state.messages = []
            st.rerun()
        if st.button(
            "Reset session",
            type="tertiary",
            icon=":material/restart_alt:",
            disabled=st.session_state.get("rag_session") is None,
            key="reset_local_session",
        ):
            try:
                close_active_rag_session()
            except Exception:
                LOGGER.exception("The local RAG session could not be reset safely.")
                st.error("The local session could not be reset. See the terminal log.")
            else:
                st.rerun()


def render_welcome():
    with st.container(border=True):
        st.markdown("### Ask your local knowledge base")
        st.caption(
            "Explore the indexed documents and inspect the evidence behind each answer."
        )
        selected = None
        for row_start in range(0, len(EXAMPLE_QUESTIONS), 2):
            columns = st.columns(2)
            for offset, column in enumerate(columns):
                index = row_start + offset
                if index >= len(EXAMPLE_QUESTIONS):
                    continue
                question = EXAMPLE_QUESTIONS[index]
                if column.button(
                    question,
                    key=f"chat_example_{index}",
                    width="stretch",
                ):
                    selected = question
        return selected


def render_chat_page():
    st.markdown(
        """
        <div class="fabric-aurora-hero">
            <div class="aurora-mark" aria-hidden="true">
                <span class="aurora-node aurora-node-coral"></span>
                <span class="aurora-node aurora-node-mint"></span>
                <span class="aurora-node aurora-node-lilac"></span>
            </div>
            <div class="aurora-hero-copy">
                <h1>Fabric Local Knowledge Assistant</h1>
                <p class="hero-primary">Private, cited answers from your documents.</p>
                <p>Local-first retrieval and generation with Microsoft Foundry Local.</p>
                <span class="aurora-badge">LOCAL · PRIVATE · CITED</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_health_metrics()

    selected_example = None
    if not st.session_state.messages:
        selected_example = render_welcome()
    else:
        for message in st.session_state.messages:
            render_message(message)

    render_chat_controls()
    st.markdown(
        '<div class="chat-input-clearance" aria-hidden="true"></div>',
        unsafe_allow_html=True,
    )
    typed_question = st.chat_input("Ask a question about your documents...")
    question = selected_example if selected_example is not None else typed_question
    if question is not None:
        question = question.strip()
        if question:
            handle_question(question)
            st.rerun()
        else:
            st.warning("Please enter a question.")
