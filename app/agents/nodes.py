from __future__ import annotations

import re
from collections.abc import Callable

from langchain_core.documents import Document
from langchain_groq import ChatGroq

from app.agents.state import AgentState
from app.config import Settings
from app.llm.groq_client import llm_text
from app.prompts.self_rag import (
    CODE_FOCUS_PROMPT,
    GENERATION_PROMPT,
    GROUNDEDNESS_PROMPT,
    REFINE_PROMPT,
    RESEARCH_FOCUS_PROMPT,
    UTILITY_PROMPT,
)
from app.retrieval.faiss_store import FaissVectorStore
from app.retrieval.reranker import CrossEncoderReranker


CODE_QUERY_TERMS = {
    "code",
    "implement",
    "implementation",
    "pseudocode",
    "script",
    "snippet",
}


CODE_QUERY_PHRASES = {
    "build a function",
    "give me code",
    "implement this",
    "python code",
    "write code",
    "write a script",
}


BROAD_DOCUMENT_TERMS = {
    "all",
    "brief",
    "complete",
    "detail",
    "detailed",
    "entire",
    "full",
    "key",
    "overview",
    "points",
    "summarize",
    "summary",
    "whole",
}


def _format_context(chunks: list[Document]) -> str:
    if not chunks:
        return "No context available."
    blocks: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source_file", "unknown")
        page = chunk.metadata.get("page", "?")
        hybrid = chunk.metadata.get("hybrid_score")
        score = f" hybrid_score={hybrid:.3f}" if isinstance(hybrid, float) else ""
        blocks.append(
            f"[{idx}] source={source} page={page}{score}\n{chunk.page_content.strip()}"
        )
    return "\n\n".join(blocks)


def _normalize_token(text: str) -> str:
    tokens = text.strip().lower().split()
    if not tokens:
        return ""
    return tokens[0].strip(".,!?:;")


def _classify_query(question: str, llm_label: str) -> str:
    normalized = question.lower()
    words = _question_words(question)

    if words & CODE_QUERY_TERMS:
        return "code"
    if any(phrase in normalized for phrase in CODE_QUERY_PHRASES):
        return "code"
    return "research"


def _is_broad_document_query(question: str) -> bool:
    words = _question_words(question)
    if words & BROAD_DOCUMENT_TERMS:
        return True
    normalized = question.lower()
    return any(
        phrase in normalized
        for phrase in (
            "about this document",
            "about this pdf",
            "all details",
            "key points",
            "main points",
            "tell me about",
        )
    )


def _expand_retrieval_query(question: str) -> str:
    terms = _question_words(question)
    expansions: set[str] = set()
    for term in terms:
        expansions.add(term)
        expansions.add(_simple_stem(term))

    if not expansions:
        return question
    return f"{question} {' '.join(sorted(expansions))}"


def _question_words(question: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", question.lower().replace("'s", "")))


def _simple_stem(term: str) -> str:
    if len(term) > 5 and term.endswith("ships"):
        return term[:-5]
    if len(term) > 4 and term.endswith("ship"):
        return term[:-4]
    if len(term) > 5 and term.endswith("ies"):
        return f"{term[:-3]}y"
    if len(term) > 5 and term.endswith("ing"):
        return term[:-3]
    if len(term) > 4 and term.endswith("ed"):
        return term[:-2]
    if len(term) > 3 and term.endswith("s"):
        return term[:-1]
    return term


class SelfRagNodes:
    def __init__(
        self,
        settings: Settings,
        llm: ChatGroq,
        vector_store: FaissVectorStore | None,
        reranker_factory: Callable[[], CrossEncoderReranker] | None,
    ):
        self.settings = settings
        self.llm = llm
        self.vector_store = vector_store
        self.reranker_factory = reranker_factory
        self._reranker: CrossEncoderReranker | None = None

    def classify_query(self, state: AgentState) -> dict:
        query_type = _classify_query(state["question"], "")
        return {
            "query_type": query_type,
            "trace": [f"Query classified as: {query_type}"],
        }

    def decide_retrieve(self, state: AgentState) -> dict:
        has_index = self.vector_store is not None and self.vector_store.document_count > 0
        if not has_index:
            return {
                "retrieve_decision": "noretrieve",
                "trace": ["Retrieve decision: noretrieve (no indexed documents)."],
            }

        return {
            "retrieve_decision": "retrieve",
            "trace": ["Retrieve decision: retrieve (indexed documents available)."],
        }

    def retrieve(self, state: AgentState) -> dict:
        if state["retrieve_decision"] != "retrieve" or self.vector_store is None:
            return {"retrieved_chunks": [], "trace": ["Skipped retrieval."]}

        k = min(max(self.settings.top_k_retrieve, self.settings.top_k_rerank), self.vector_store.document_count)
        if _is_broad_document_query(state["question"]):
            k = min(max(k, self.settings.top_k_retrieve), self.vector_store.document_count)
        retrieval_query = _expand_retrieval_query(state["question"])
        if self.settings.hybrid_search:
            chunks = self.vector_store.hybrid_search(
                retrieval_query,
                k=k,
                dense_weight=self.settings.hybrid_dense_weight,
                lexical_candidate_pool=self.settings.lexical_candidate_pool,
            )
            mode = "hybrid FAISS/BM25"
        else:
            chunks = self.vector_store.similarity_search(
                retrieval_query, k=k
            )
            mode = "FAISS"
        return {
            "retrieved_chunks": chunks,
            "trace": [f"Retrieved {len(chunks)} chunks with {mode} search."],
        }

    def rerank(self, state: AgentState) -> dict:
        if not state["retrieved_chunks"]:
            return {"reranked_chunks": [], "trace": ["Skipped reranking."]}
        if _is_broad_document_query(state["question"]):
            chunks = state["retrieved_chunks"][: self.settings.top_k_retrieve]
            return {
                "reranked_chunks": chunks,
                "trace": [f"Kept {len(chunks)} chunks for broad document question."],
            }

        reranker = self._get_reranker()
        if reranker is None:
            return {
                "reranked_chunks": state["retrieved_chunks"][: self.settings.top_k_rerank],
                "trace": ["Skipped reranking because no reranker is configured."],
            }

        reranked = reranker.rerank(state["question"], state["retrieved_chunks"])
        return {
            "reranked_chunks": reranked,
            "trace": [f"Reranked to top {len(reranked)} chunks."],
        }

    def _get_reranker(self) -> CrossEncoderReranker | None:
        if self.reranker_factory is None:
            return None
        if self._reranker is None:
            self._reranker = self.reranker_factory()
        return self._reranker

    def filter_relevance(self, state: AgentState) -> dict:
        filtered = state["reranked_chunks"]
        return {
            "relevance_labels": ["kept reranked context"],
            "filtered_chunks": filtered,
            "trace": [f"Kept {len(filtered)} reranked chunks as context."],
        }

    def generate(self, state: AgentState) -> dict:
        context = _format_context(state["filtered_chunks"])
        if state["query_type"] == "code":
            prompt = CODE_FOCUS_PROMPT.format(
                question=state["question"], context=context
            )
        else:
            prompt = RESEARCH_FOCUS_PROMPT.format(
                question=state["question"], context=context
            )

        if not state["filtered_chunks"]:
            prompt = GENERATION_PROMPT.format(
                question=state["question"],
                query_type=state["query_type"],
                context=context,
            )

        answer = llm_text(self.llm, prompt)
        return {"draft_answer": answer, "trace": ["Generated draft answer."]}

    def verify_groundedness(self, state: AgentState) -> dict:
        context = _format_context(state["filtered_chunks"])
        raw = llm_text(
            self.llm,
            GROUNDEDNESS_PROMPT.format(
                question=state["question"],
                context=context,
                answer=state["draft_answer"],
            ),
        )
        label = _normalize_token(raw)
        groundedness = "supported" if "supported" in label and "unsupported" not in label else "unsupported"
        issues = state.get("issues", [])
        if groundedness == "unsupported":
            issues = issues + ["Answer contains unsupported factual claims."]
        return {
            "groundedness": groundedness,
            "issues": issues,
            "trace": [f"Groundedness check: {groundedness}"],
        }

    def verify_utility(self, state: AgentState) -> dict:
        raw = llm_text(
            self.llm,
            UTILITY_PROMPT.format(
                question=state["question"],
                query_type=state["query_type"],
                answer=state["draft_answer"],
            ),
        )
        label = _normalize_token(raw)
        utility = "useful" if "useful" in label and "notuseful" not in label else "notuseful"
        issues = state.get("issues", [])
        if utility == "notuseful":
            issues = issues + ["Answer is incomplete or not useful for the query type."]
        return {
            "utility": utility,
            "issues": issues,
            "trace": [f"Utility check: {utility}"],
        }

    def refine(self, state: AgentState) -> dict:
        context = _format_context(state["filtered_chunks"])
        issues = "\n".join(f"- {item}" for item in state.get("issues", []))
        revised = llm_text(
            self.llm,
            REFINE_PROMPT.format(
                question=state["question"],
                query_type=state["query_type"],
                context=context,
                answer=state["draft_answer"],
                issues=issues or "- Improve clarity and grounding.",
            ),
        )
        return {
            "draft_answer": revised,
            "iteration": state.get("iteration", 0) + 1,
            "issues": [],
            "trace": [f"Refined answer at iteration {state.get('iteration', 0) + 1}."],
        }

    def finalize(self, state: AgentState) -> dict:
        return {
            "final_answer": state["draft_answer"],
            "trace": ["Self-RAG pipeline completed."],
        }
