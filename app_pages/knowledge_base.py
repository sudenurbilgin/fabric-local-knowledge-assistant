import logging
from collections import Counter

import altair as alt
import streamlit as st

from fabric_rag.ingestion import SUPPORTED_UPLOAD_TYPES
from fabric_rag.knowledge_base import build_knowledge_base
from fabric_rag.management import (
    get_indexed_passage_counts_by_source,
    get_source_file_details,
    remove_source_document,
    save_source_document,
)

from .shared import (
    close_active_rag_session,
    format_file_size,
    friendly_source_title,
    indexed_counts,
    read_management_status,
    render_notice,
    set_notice,
    source_type_label,
)


LOGGER = logging.getLogger(__name__)
NOTICE_KEY = "knowledge_base_notice"
REBUILD_STAGES = {
    "preparing_documents": "Preparing documents...",
    "extracting_text": "Extracting and cleaning text...",
    "generating_embeddings": "Generating local embeddings...",
    "rebuilding_local_index": "Rebuilding the local index...",
    "validating": "Validating the rebuilt index...",
    "complete": "Complete.",
}


@st.dialog("Remove source document", width="small")
def confirm_source_removal(filename):
    st.write(f"Remove **{filename}** from the source collection?")
    st.caption(
        "The currently indexed knowledge base will remain unchanged until you "
        "explicitly rebuild it."
    )
    cancel_column, remove_column = st.columns(2)
    if cancel_column.button("Cancel", width="stretch"):
        st.rerun()
    if remove_column.button(
        "Remove document",
        type="primary",
        width="stretch",
    ):
        try:
            removed = remove_source_document(filename)
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
            st.error(str(error))
        else:
            set_notice(
                NOTICE_KEY,
                "success",
                f"Removed {removed}. Rebuild the index to apply this change.",
            )
            st.rerun()


def render_metrics(status):
    indexed_documents, indexed_passages = indexed_counts(status)
    index_status = (
        "Unavailable"
        if status is None
        else "Rebuild required"
        if status["rebuild_required"]
        else "Up to date"
    )
    source_count = "—" if status is None else status["source_document_count"]

    columns = st.columns(4)
    columns[0].metric(
        "Source files",
        source_count,
        border=True,
        icon=":material/folder_open:",
    )
    columns[1].metric(
        "Indexed documents",
        indexed_documents if indexed_documents is not None else "—",
        border=True,
        icon=":material/description:",
    )
    columns[2].metric(
        "Indexed passages",
        indexed_passages if indexed_passages is not None else "—",
        border=True,
        icon=":material/segment:",
    )
    columns[3].metric(
        "Index status",
        index_status,
        border=True,
        icon=(
            ":material/warning:"
            if index_status == "Rebuild required"
            else ":material/check_circle:"
        ),
    )


def render_charts():
    source_details = get_source_file_details()
    file_type_counts = Counter(
        source_type_label(detail["filename"])
        for detail in source_details
    )
    passage_chart = [
        {
            "Document": friendly_source_title(source),
            "Indexed passages": count,
        }
        for source, count in get_indexed_passage_counts_by_source()
    ]

    st.markdown("### Current source collection")
    with st.container(border=True):
        st.caption("Supported formats")
        st.markdown("**MD · TXT · PDF · DOCX**")
        current_types = " · ".join(sorted(file_type_counts)) or "None"
        st.caption(f"Current source collection: {current_types}")

    if len(file_type_counts) > 1:
        type_chart = [
            {"File type": file_type, "Documents": count}
            for file_type, count in sorted(file_type_counts.items())
        ]
        st.markdown("#### Current source formats")
        st.bar_chart(
            type_chart,
            x="File type",
            y="Documents",
            height=190,
        )

    st.markdown("### Indexed passages by source")
    largest_passage_count = max(
        (item["Indexed passages"] for item in passage_chart),
        default=0,
    )
    passage_base = alt.Chart(alt.Data(values=passage_chart)).encode(
        x=alt.X(
            "Indexed passages:Q",
            title="Indexed passages",
            scale=alt.Scale(domain=[0, largest_passage_count + 3]),
        ),
        y=alt.Y(
            "Document:N",
            title=None,
            sort="-x",
            axis=alt.Axis(labelLimit=280),
        ),
        tooltip=(
            alt.Tooltip("Document:N", title="Document"),
            alt.Tooltip("Indexed passages:Q", title="Passages"),
        ),
    )
    passage_bars = passage_base.mark_bar(
        color="#E98A77",
        cornerRadiusEnd=4,
    )
    passage_labels = passage_base.mark_text(
        align="left",
        baseline="middle",
        dx=6,
        color="#F4EEE8",
    ).encode(text=alt.Text("Indexed passages:Q"))
    st.altair_chart(
        (passage_bars + passage_labels).properties(height=360),
        width="stretch",
    )


def render_current_sources(status):
    st.markdown("### Current sources")
    details = get_source_file_details()
    rows = [
        {
            "Document": detail["filename"],
            "Type": source_type_label(detail["filename"]),
            "Size": format_file_size(detail["size_bytes"]),
            "State": format_source_state(
                status["source_states"].get(
                    detail["filename"],
                    "Pending addition",
                )
            ),
        }
        for detail in details
    ]
    st.dataframe(
        rows,
        hide_index=True,
        width="stretch",
        column_order=("Document", "Type", "Size", "State"),
        column_config={
            "Document": st.column_config.TextColumn(width="large"),
            "Type": st.column_config.TextColumn(width="medium"),
            "Size": st.column_config.TextColumn(width="small"),
            "State": st.column_config.TextColumn(width="medium"),
        },
        row_height=34,
    )

    selected = st.selectbox(
        "Document to remove",
        status["source_files"],
        key="source_removal_selection",
    )
    final_source = len(status["source_files"]) <= 1
    if st.button(
        "Remove selected document",
        disabled=final_source,
    ):
        confirm_source_removal(selected)
    if final_source:
        st.caption("The final usable source document cannot be removed.")


def format_source_state(state):
    if state == "Indexed":
        return "✓ Indexed"
    if state in {"Pending addition", "Pending update"}:
        return f"△ {state}"
    return f"− {state}"


def render_upload():
    st.markdown("### Add documents")
    st.caption("Supported formats: Markdown, plain text, text-based PDF, and DOCX.")
    uploads = st.file_uploader(
        "Choose source documents",
        type=list(SUPPORTED_UPLOAD_TYPES),
        accept_multiple_files=True,
        key="knowledge_base_uploads",
    )
    if not st.button(
        "Add to source collection",
        disabled=not uploads,
        type="primary",
    ):
        return

    added = []
    errors = []
    for upload in uploads:
        try:
            added.append(save_source_document(upload.name, upload.getvalue()))
        except (FileExistsError, OSError, TypeError, ValueError) as error:
            errors.append(f"{upload.name}: {error}")

    if added:
        message = (
            f"Added {len(added)} source document(s): {', '.join(added)}. "
            "Rebuild the index to use them in answers."
        )
        if errors:
            message += " Not added: " + " | ".join(errors)
            set_notice(NOTICE_KEY, "warning", message)
        else:
            set_notice(NOTICE_KEY, "success", message)
        st.rerun()

    for error in errors:
        st.error(error)


def render_pending_status(status):
    st.markdown("### Pending changes and index status")
    if status["rebuild_required"]:
        st.warning(
            "Rebuild required. The current chat remains available using the "
            "previous valid index."
        )
    else:
        st.success("The source collection and local index are up to date.")

    if status["pending_removed_sources"]:
        st.caption("Removed sources still present in the current index:")
        for filename in status["pending_removed_sources"]:
            st.text(filename)


def render_rebuild(status):
    st.markdown("### Rebuild index")
    st.caption(
        "A rebuild extracts the current sources, generates local embeddings, "
        "and transactionally replaces the SQLite index."
    )
    if not st.button(
        "Rebuild knowledge base",
        type="primary",
        disabled=not status["rebuild_required"],
    ):
        return

    with st.status("Rebuilding knowledge base...", expanded=True) as rebuild_status:
        try:
            close_active_rag_session()
            result = build_knowledge_base(
                progress_callback=lambda stage: rebuild_status.write(
                    REBUILD_STAGES[stage]
                )
            )
        except Exception as error:
            LOGGER.exception("Knowledge-base rebuild failed.")
            rebuild_status.update(label="Rebuild failed", state="error")
            st.error("Rebuild failed. The previous valid index has been preserved.")
            with st.expander("Developer details"):
                st.code(str(error))
            return

        rebuild_status.update(
            label="Knowledge base updated",
            state="complete",
            expanded=False,
        )
        set_notice(
            NOTICE_KEY,
            "success",
            "Knowledge base updated. The next question will use the new index. "
            f"{result['statistics']['document_count']} documents and "
            f"{result['statistics']['chunk_count']} passages are indexed.",
        )
        st.rerun()


def render_knowledge_base_page():
    st.title("Knowledge Base")
    st.markdown(
        "Manage local source documents, review index coverage, and apply staged "
        "changes with an explicit rebuild."
    )
    render_notice(NOTICE_KEY)
    status = read_management_status()
    render_metrics(status)
    if status is None:
        st.error("Knowledge-base management status is currently unavailable.")
        return

    render_charts()
    st.divider()
    render_current_sources(status)
    render_upload()
    render_pending_status(status)
    render_rebuild(status)
