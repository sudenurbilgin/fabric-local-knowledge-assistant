import ast
import importlib.util
from pathlib import Path

from foundry_local_sdk import FoundryLocalManager
from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROUTER_PATH = PROJECT_ROOT / "streamlit_app.py"


def metric_values(app_test):
    return {metric.label: metric.value for metric in app_test.metric}


def navigation_definition():
    tree = ast.parse(ROUTER_PATH.read_text(encoding="utf-8"))
    page_titles = []
    top_navigation = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr == "Page":
            for keyword in node.keywords:
                if keyword.arg == "title" and isinstance(keyword.value, ast.Constant):
                    page_titles.append(keyword.value.value)
        if node.func.attr == "navigation":
            top_navigation = any(
                keyword.arg == "position"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == "top"
                for keyword in node.keywords
            )
    return page_titles, top_navigation


def run_page(script):
    return AppTest.from_string(script, default_timeout=20).run()


def import_file(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_chat_controls():
    script = """
import streamlit as st
from app_pages.chat import render_chat_page

class FakeSession:
    def close(self):
        st.session_state.close_calls += 1

if "control_test_initialized" not in st.session_state:
    st.session_state.control_test_initialized = True
    st.session_state.close_calls = 0
    st.session_state.rag_session = FakeSession()
    st.session_state.messages = [{"role": "user", "content": "Keep me until clear."}]

render_chat_page()
"""
    chat = AppTest.from_string(script, default_timeout=20).run()
    clear_button = next(button for button in chat.button if button.label == "Clear")
    chat = clear_button.click().run()
    clear_preserved_session = (
        not chat.exception
        and chat.session_state.close_calls == 0
        and "rag_session" in chat.session_state
        and chat.session_state.messages == []
    )

    reset_button = next(
        button for button in chat.button if button.label == "Reset session"
    )
    chat = reset_button.click().run()
    reset_metric_values = metric_values(chat)
    reset_closed_only_session = (
        not chat.exception
        and chat.session_state.close_calls == 1
        and "rag_session" not in chat.session_state
        and chat.session_state.messages == []
        and reset_metric_values.get("Local inference") == "Ready on demand"
    )
    return clear_preserved_session, reset_closed_only_session


def main():
    results = {}
    manager_before = FoundryLocalManager.instance

    page_titles, top_navigation = navigation_definition()
    results["Top navigation defines exactly three primary pages"] = (
        top_navigation
        and page_titles == ["Chat", "Knowledge Base", "System & Evaluation"]
    )

    chat = AppTest.from_file(ROUTER_PATH, default_timeout=20).run()
    chat_metrics = metric_values(chat)
    results["Chat page renders without an exception"] = (
        not chat.exception
        and any(
            "fabric-aurora-hero" in element.value
            and "Fabric Local Knowledge Assistant" in element.value
            for element in chat.markdown
        )
    )
    results["Chat page renders current health metrics"] = (
        chat_metrics.get("Indexed sources") == "8"
        and chat_metrics.get("Indexed passages") == "90"
        and chat_metrics.get("Knowledge base") == "Up to date"
        and chat_metrics.get("Local inference") == "Ready on demand"
    )
    clear_preserved_session, reset_closed_only_session = validate_chat_controls()
    results["Clear chat preserves the active local session"] = clear_preserved_session
    results["Reset session closes only the session and restores idle status"] = (
        reset_closed_only_session
    )
    results["Chat controls and input render together at conversation bottom"] = (
        any(button.label == "Clear" for button in chat.button)
        and any(button.label == "Reset session" for button in chat.button)
        and len(chat.chat_input) == 1
    )

    knowledge_base = run_page(
        "from app_pages.knowledge_base import render_knowledge_base_page\n"
        "render_knowledge_base_page()"
    )
    knowledge_metrics = metric_values(knowledge_base)
    results["Knowledge Base page renders current status"] = (
        not knowledge_base.exception
        and knowledge_metrics.get("Source files") == "8"
        and knowledge_metrics.get("Indexed documents") == "8"
        and knowledge_metrics.get("Indexed passages") == "90"
        and knowledge_metrics.get("Index status") == "Up to date"
    )
    results["Knowledge Base page renders passage chart and source table"] = (
        len(knowledge_base.get("vega_lite_chart")) >= 1
        and len(knowledge_base.dataframe) == 1
        and any(
            "MD · TXT · PDF · DOCX" in element.value
            for element in knowledge_base.markdown
        )
    )

    system = run_page(
        "from app_pages.system_evaluation import render_system_evaluation_page\n"
        "render_system_evaluation_page()"
    )
    system_metrics = metric_values(system)
    system_markdown = [element.value for element in system.markdown]
    results["System & Evaluation page renders benchmark and validation"] = (
        not system.exception
        and system_metrics.get("Measured warm speedup") == "6.06x"
        and len(system.get("vega_lite_chart")) == 1
        and any(
            "Cloud LLM API" in element.value
            for element in system.caption
        )
        and any("Top-3 retrieval regression" in value for value in system_markdown)
        and any("architecture-flow" in value for value in system_markdown)
    )

    import_file("cli_presentation_validation", PROJECT_ROOT / "cli.py")
    import_file("streamlit_presentation_validation", ROUTER_PATH)

    results["UI and CLI imports do not initialize Foundry Local"] = (
        manager_before is None and FoundryLocalManager.instance is None
    )

    print("STREAMLIT PRESENTATION MODEL-FREE VALIDATION")
    print("=" * 50)
    for name, passed in results.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    passed_count = sum(results.values())
    print(f"\nPassed: {passed_count} / {len(results)}")
    if passed_count != len(results):
        raise RuntimeError("At least one Streamlit presentation check failed.")


if __name__ == "__main__":
    main()
