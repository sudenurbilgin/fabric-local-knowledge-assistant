import altair as alt
import streamlit as st

from fabric_rag.config import (
    CHAT_MODEL_VARIANT_ID,
    CPU_EXECUTION_PROVIDER,
    DATABASE_PATH,
    EMBEDDING_MODEL_ALIAS,
    RETRIEVAL_TOP_K,
)


STATELESS_WARM_SECONDS = 2.31
PERSISTENT_WARM_SECONDS = 0.38
MEASURED_WARM_SPEEDUP = 6.06
ENGINEERING_VALIDATIONS = (
    {
        "Validation": "Top-3 retrieval regression",
        "Result": "4 / 4 PASS",
        "Evidence": "Persistent retrieval comparison",
    },
    {
        "Validation": "Knowledge Base Manager safety",
        "Result": "12 / 12 PASS",
        "Evidence": "Model-free management diagnostic",
    },
    {
        "Validation": "Multi-format source ingestion",
        "Result": "13 / 13 PASS",
        "Evidence": "Model-free ingestion diagnostic",
    },
    {
        "Validation": "Dynamic corpus validation",
        "Result": "PASS",
        "Evidence": "Temporary mixed-corpus rebuild",
    },
    {
        "Validation": "Persistent retriever reuse",
        "Result": "PASS",
        "Evidence": "Lifecycle and call-count diagnostic",
    },
)


def render_system_configuration():
    st.markdown("## System")
    first_row = st.columns(3)
    with first_row[0].container(border=True):
        st.caption("Chat model")
        st.markdown("### Phi-4 Mini")
        st.caption(CHAT_MODEL_VARIANT_ID)
    with first_row[1].container(border=True):
        st.caption("Embedding model")
        st.markdown("### Qwen3 Embedding 0.6B")
        st.caption(EMBEDDING_MODEL_ALIAS)
    with first_row[2].container(border=True):
        st.caption("Execution")
        st.markdown("### CPU · Local")
        st.caption(CPU_EXECUTION_PROVIDER)

    second_row = st.columns(3)
    with second_row[0].container(border=True):
        st.caption("Retrieval")
        st.markdown(f"### Top-{RETRIEVAL_TOP_K} cosine")
        st.caption("Cosine similarity ranking")
    with second_row[1].container(border=True):
        st.caption("Database")
        st.markdown(f"### SQLite · {DATABASE_PATH.name}")
        st.caption("Local knowledge base")
    with second_row[2].container(border=True):
        st.caption("Cloud LLM API")
        st.markdown("### None")
        st.caption("Answer generation remains local")


def render_architecture():
    st.markdown("## Architecture")
    stages = (
        "Documents",
        "Extraction & Cleaning",
        "Chunking",
        "Embeddings",
        "SQLite Knowledge Base",
        "Top-3 Retrieval",
        "Phi-4 Mini",
        "Cited Answer",
    )
    flow = '<div class="architecture-flow">'
    tones = ("tone-coral", "tone-mint", "tone-lilac")
    for index, stage in enumerate(stages):
        flow += (
            f'<div class="architecture-stage {tones[index % len(tones)]}">'
            f"<span>{stage}</span></div>"
        )
        if index < len(stages) - 1:
            flow += '<div class="architecture-arrow">→</div>'
    flow += "</div>"
    st.markdown(flow, unsafe_allow_html=True)
    st.caption(
        "All document processing, retrieval, and answer generation run through "
        "the local project pipeline."
    )


def render_benchmark():
    st.markdown("## Engineering benchmark")
    st.caption("Historical engineering benchmark on the original baseline corpus.")
    metric_column, chart_column = st.columns([1, 2.4])
    with metric_column:
        st.metric(
            "Measured warm speedup",
            f"{MEASURED_WARM_SPEEDUP:.2f}x",
            border=True,
        )
        st.caption(
            "Warm-query comparison only. Cold model initialization is excluded."
        )
    with chart_column:
        benchmark_data = [
            {
                "Retrieval path": "Persistent retriever",
                "Seconds": PERSISTENT_WARM_SECONDS,
            },
            {
                "Retrieval path": "Previous stateless",
                "Seconds": STATELESS_WARM_SECONDS,
            },
        ]
        benchmark_base = alt.Chart(alt.Data(values=benchmark_data)).encode(
            x=alt.X(
                "Seconds:Q",
                title="Average warm retrieval time (seconds)",
                scale=alt.Scale(domain=[0, 2.55]),
            ),
            y=alt.Y(
                "Retrieval path:N",
                title=None,
                sort=["Persistent retriever", "Previous stateless"],
            ),
            tooltip=(
                alt.Tooltip("Retrieval path:N", title="Retrieval path"),
                alt.Tooltip("Seconds:Q", title="Seconds", format=".2f"),
            ),
        )
        benchmark_bars = benchmark_base.mark_bar(cornerRadiusEnd=4).encode(
            color=alt.Color(
                "Retrieval path:N",
                legend=None,
                scale=alt.Scale(
                    domain=["Persistent retriever", "Previous stateless"],
                    range=["#79CEB2", "#D97C6D"],
                ),
            )
        )
        benchmark_labels = benchmark_base.transform_calculate(
            display_time="format(datum.Seconds, '.2f') + ' s'"
        ).mark_text(
            align="left",
            baseline="middle",
            dx=7,
            color="#F4EEE8",
        ).encode(text="display_time:N")
        st.altair_chart(
            (benchmark_bars + benchmark_labels).properties(height=165),
            width="stretch",
        )
    st.caption(
        "Previous stateless warm retrieval: approximately 2.31 seconds · "
        "Persistent retrieval: approximately 0.38 seconds. Individual queries vary."
    )


def render_validation():
    st.markdown("## Engineering validation")
    st.caption(
        "Recorded project diagnostics—not live answer-quality, accuracy, or "
        "confidence scores."
    )
    with st.container(border=True):
        for index, validation in enumerate(ENGINEERING_VALIDATIONS):
            st.markdown(
                f":green[**✓**] **{validation['Validation']}** — "
                f"{validation['Result'].replace(' PASS', '')}"
            )
            st.caption(validation["Evidence"])
            if index < len(ENGINEERING_VALIDATIONS) - 1:
                st.markdown(":gray[────────]")


def render_system_evaluation_page():
    st.title("System & Evaluation")
    st.markdown(
        "A concise view of the validated local architecture, engineering "
        "benchmark, and regression coverage."
    )
    render_system_configuration()
    render_architecture()
    render_benchmark()
    render_validation()

    st.markdown("## Known limitation")
    st.warning(
        "Broad questions whose required evidence spans chunk boundaries may not "
        "receive every required passage within the fixed Top-3 context."
    )
