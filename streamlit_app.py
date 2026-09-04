import streamlit as st

from app_pages.chat import render_chat_page
from app_pages.knowledge_base import render_knowledge_base_page
from app_pages.shared import apply_visual_polish, initialize_state
from app_pages.system_evaluation import render_system_evaluation_page


def main():
    st.set_page_config(
        page_title="Fabric Local Knowledge Assistant",
        page_icon=":material/library_books:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    initialize_state()
    apply_visual_polish()

    page = st.navigation(
        [
            st.Page(
                render_chat_page,
                title="Chat",
                icon=":material/chat:",
                url_path="chat",
                default=True,
            ),
            st.Page(
                render_knowledge_base_page,
                title="Knowledge Base",
                icon=":material/library_books:",
                url_path="knowledge-base",
            ),
            st.Page(
                render_system_evaluation_page,
                title="System & Evaluation",
                icon=":material/monitoring:",
                url_path="system-evaluation",
            ),
        ],
        position="top",
    )
    page.run()


if __name__ == "__main__":
    main()
