from __future__ import annotations

import sys
import shutil
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import FAISS_DIR, Settings, ensure_data_dirs
from app.pipeline import SelfRagPipeline


st.set_page_config(
    page_title="Self-RAG PDF Q&A",
    page_icon=":material/library_books:",
    layout="wide",
)

ensure_data_dirs()


@st.cache_resource
def get_pipeline(settings: Settings) -> SelfRagPipeline:
    return SelfRagPipeline(settings)


def has_saved_index() -> bool:
    return FAISS_DIR.exists() and any(FAISS_DIR.iterdir())


def clear_saved_index() -> None:
    if FAISS_DIR.exists():
        shutil.rmtree(FAISS_DIR)
    ensure_data_dirs()
    st.session_state.pop("indexed_chunks", None)


def main() -> None:
    st.title("Self-RAG PDF Q&A")
    st.caption(
        "Upload PDFs, ask code or research questions, and get answers verified by Self-RAG prompts."
    )

    try:
        settings = Settings.from_env()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    if not settings.groq_api_key:
        st.error("Set `GROQ_API_KEY` in `.env` before running queries.")
        st.code("copy .env.example .env")

    with st.sidebar:
        st.subheader("Index status")
        if "indexed_chunks" in st.session_state:
            st.metric("Indexed chunks", st.session_state.indexed_chunks)
        else:
            st.metric("Saved index", "Yes" if has_saved_index() else "No")
        st.markdown("**Stack**")
        st.markdown("- Semantic chunking")
        st.markdown("- Hybrid FAISS/BM25 retrieval")
        st.markdown("- Cross-encoder reranking")
        st.markdown("- Groq LLM")
        st.markdown("- LangGraph Self-RAG agents")
        st.markdown("**Retrieval settings**")
        st.write(f"Top-K retrieve: {settings.top_k_retrieve}")
        st.write(f"Top-K rerank: {settings.top_k_rerank}")
        st.write(f"Dense weight: {settings.hybrid_dense_weight:.2f}")
        st.write(f"Max upload batch: {settings.max_upload_files}")
        if st.button("Clear index", type="secondary"):
            clear_saved_index()
            st.success("Cleared the FAISS index. Re-index your PDFs before asking questions.")
            st.rerun()

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("1) Upload PDFs")
        uploads = st.file_uploader(
            "Select one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True,
        )
        too_many_uploads = bool(uploads) and len(uploads) > settings.max_upload_files
        if too_many_uploads:
            st.warning(f"Upload at most {settings.max_upload_files} PDFs in one batch.")

        if st.button("Index PDFs", type="primary", disabled=not uploads or too_many_uploads):
            with st.spinner("Parsing, chunking, embedding, and indexing..."):
                try:
                    pipeline = get_pipeline(settings)
                    summary = pipeline.index_uploaded_pdfs(uploads)
                except Exception as exc:
                    st.error(f"Indexing failed: {exc}")
                    st.stop()

            st.session_state.indexed_chunks = summary["total_chunks"]
            st.success(
                f"Indexed {summary['chunks_added']} chunks from {len(summary['files'])} file(s)."
            )
            for warning in summary.get("warnings", []):
                st.warning(warning)
            st.write("Files:", ", ".join(summary["files"]))
            for item in summary["preview"]:
                label = f"{item['source']} (page {item['page']}, {item['strategy']})"
                with st.expander(label):
                    st.write(item["text"])

    with col_right:
        st.subheader("2) Ask a question")
        question = st.text_area(
            "Code or research question",
            placeholder="Example: Implement the algorithm described in section 3 and explain complexity.",
            height=120,
        )
        can_query = bool(question.strip()) and bool(settings.groq_api_key)
        if st.button("Run Self-RAG", type="primary", disabled=not can_query):
            with st.spinner("Running retrieval, reranking, generation, and Self-RAG verification..."):
                try:
                    pipeline = get_pipeline(settings)
                    st.session_state.indexed_chunks = pipeline.indexed_chunks
                    result = pipeline.ask(question.strip())
                except Exception as exc:
                    st.error(f"Self-RAG failed: {exc}")
                    st.stop()

            st.markdown("### Answer")
            st.markdown(result["answer"])

            st.markdown("### Verification")
            v1, v2, v3, v4 = st.columns(4)
            v1.metric("Query type", result["query_type"])
            v2.metric("Retrieve", result["retrieve_decision"])
            v3.metric("Groundedness", result["groundedness"])
            v4.metric("Utility", result["utility"])

            if result["citations"]:
                st.markdown("### Citations")
                for cite in result["citations"]:
                    st.write(f"- {cite['source']} (page {cite['page']})")

            if result["supporting_chunks"]:
                with st.expander("Supporting context"):
                    for item in result["supporting_chunks"]:
                        st.markdown(f"**{item['label']}**")
                        st.write(item["text"])

            with st.expander("Self-RAG trace"):
                for line in result["trace"]:
                    st.write(f"- {line}")


if __name__ == "__main__":
    main()
