from __future__ import annotations

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document

from app.config import Settings
from app.agents.nodes import (
    SelfRagNodes,
    _classify_query,
    _expand_retrieval_query,
    _is_broad_document_query,
)
from app.prompts.self_rag import RESEARCH_FOCUS_PROMPT
from app.ingestion.chunker import chunk_documents
from app.pipeline import _safe_pdf_filename
from app.retrieval.faiss_store import FaissVectorStore


def make_settings(**overrides) -> Settings:
    values = {
        "groq_api_key": "",
        "groq_model": "llama-3.3-70b-versatile",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "semantic_chunking": True,
        "semantic_min_chunk_size": 40,
        "semantic_similarity_threshold": 0.5,
        "hybrid_search": True,
        "hybrid_dense_weight": 0.65,
        "lexical_candidate_pool": 50,
        "top_k_retrieve": 30,
        "top_k_rerank": 10,
        "max_self_rag_iterations": 2,
        "max_upload_files": 20,
    }
    values.update(overrides)
    return Settings(**values)


class FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            if "accuracy" in text.lower() or "result" in text.lower():
                vectors.append([0.0, 1.0])
            else:
                vectors.append([1.0, 0.0])
        return vectors


def test_safe_pdf_filename_strips_paths_and_symbols() -> None:
    assert _safe_pdf_filename("../my paper!.pdf") == "my_paper.pdf"
    assert _safe_pdf_filename("...pdf") == "uploaded.pdf"


def test_semantic_chunking_skips_empty_docs() -> None:
    settings = make_settings()
    chunks = chunk_documents(
        [Document(page_content="   ", metadata={"source_file": "x.pdf", "page": 1})],
        settings,
        FakeEmbeddings(),
    )
    assert chunks == []


def test_semantic_chunking_adds_metadata() -> None:
    settings = make_settings()
    document = Document(
        page_content=(
            "This section describes the method in detail. "
            "The method uses dense retrieval for evidence.\n\n"
            "The result section reports accuracy. "
            "Accuracy is compared with baselines."
        ),
        metadata={"source_file": "x.pdf", "page": 1},
    )
    chunks = chunk_documents([document], settings, FakeEmbeddings())
    assert chunks
    assert all(chunk.metadata["chunking_strategy"] == "semantic" for chunk in chunks)
    assert all("chunk_id" in chunk.metadata for chunk in chunks)


def test_lexical_scores_find_exact_terms() -> None:
    store = FaissVectorStore(
        type(
            "Store",
            (),
            {
                "docstore": InMemoryDocstore(
                    {
                        "a": Document(
                            page_content="faiss vector search retrieval",
                            metadata={"source_file": "a.pdf", "page": 1, "chunk_id": 0},
                        ),
                        "b": Document(
                            page_content="unrelated cooking notes",
                            metadata={"source_file": "b.pdf", "page": 1, "chunk_id": 1},
                        ),
                    }
                )
            },
        )()
    )
    scores = store._lexical_scores("vector retrieval", store._documents_by_key())
    assert len(scores) == 1
    assert next(iter(scores.values())) > 0


def test_indexed_documents_always_retrieve() -> None:
    settings = make_settings()
    vector_store = type("VectorStore", (), {"document_count": 1})()
    rag_nodes = SelfRagNodes(
        settings=settings,
        llm=object(),
        vector_store=vector_store,
        reranker_factory=None,
    )

    result = rag_nodes.decide_retrieve(
        {
            "question": "what are my skills?",
            "query_type": "research",
            "retrieve_decision": "noretrieve",
            "retrieved_chunks": [],
            "reranked_chunks": [],
            "filtered_chunks": [],
            "draft_answer": "",
            "final_answer": "",
            "relevance_labels": [],
            "groundedness": "unsupported",
            "utility": "notuseful",
            "issues": [],
            "iteration": 0,
            "trace": [],
        }
    )

    assert result["retrieve_decision"] == "retrieve"


def test_research_prompt_keeps_sources_out_of_answer() -> None:
    rendered = RESEARCH_FOCUS_PROMPT.format(
        question="What are my skills?",
        context="[1] source=resume.pdf page=0\nPython, SQL, machine learning",
    )
    assert "Do not include source names" in rendered
    assert "Use only facts explicitly present" in rendered
    assert "Suggested follow-up" not in rendered


def test_profile_projects_are_textual_even_if_llm_says_code() -> None:
    assert _classify_query("what projects have I built?", "code") == "research"


def test_explicit_code_question_stays_code() -> None:
    assert _classify_query("write Python code for this algorithm", "research") == "code"


def test_internship_question_is_profile_query_even_if_llm_says_code() -> None:
    assert _classify_query("has he done any internship?", "code") == "research"


def test_retrieval_query_expands_suffix_variants() -> None:
    expanded = _expand_retrieval_query("has he done any internship?")
    assert "intern" in expanded
    assert "internship" in expanded


def test_broad_document_query_detected() -> None:
    assert _is_broad_document_query("summarize the whole PDF in detail")


def test_domain_neutral_questions_are_textual() -> None:
    assert _classify_query("what APIs are mentioned in this PDF?", "code") == "research"
