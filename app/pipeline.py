from __future__ import annotations

import re
import shutil
from pathlib import Path

from langchain_core.documents import Document

from app.agents.graph import build_self_rag_graph
from app.agents.nodes import SelfRagNodes
from app.config import FAISS_DIR, Settings, UPLOAD_DIR, ensure_data_dirs
from app.ingestion.chunker import chunk_preview, ingest_pdf_paths
from app.llm.groq_client import build_llm
from app.retrieval.embeddings import build_embeddings
from app.retrieval.faiss_store import FaissVectorStore
from app.retrieval.reranker import CrossEncoderReranker


class SelfRagPipeline:
    def __init__(self, settings: Settings):
        ensure_data_dirs()
        self.settings = settings
        self.embeddings = build_embeddings(settings)
        self.llm = None
        self.reranker = None
        self.vector_store = FaissVectorStore.load(self.embeddings)

    @property
    def indexed_chunks(self) -> int:
        if self.vector_store is None:
            return 0
        return self.vector_store.document_count

    def index_uploaded_pdfs(self, uploaded_files: list) -> dict:
        if not uploaded_files:
            return {
                "files": [],
                "chunks_added": 0,
                "total_chunks": self.indexed_chunks,
                "preview": [],
                "warnings": ["No PDFs were provided."],
            }
        if len(uploaded_files) > self.settings.max_upload_files:
            raise ValueError(
                f"Upload at most {self.settings.max_upload_files} PDFs in one batch."
            )

        saved_paths: list[Path] = []
        for uploaded in uploaded_files:
            filename = _safe_pdf_filename(uploaded.name)
            target = _unique_upload_path(UPLOAD_DIR / filename)
            target.write_bytes(uploaded.getbuffer())
            saved_paths.append(target)

        chunks = ingest_pdf_paths(saved_paths, self.settings, self.embeddings)
        if not chunks:
            return {
                "files": [path.name for path in saved_paths],
                "chunks_added": 0,
                "total_chunks": self.indexed_chunks,
                "preview": [],
                "warnings": [
                    "No extractable text was found. Scanned PDFs need OCR before indexing."
                ],
            }

        if self.vector_store is None:
            self.vector_store = FaissVectorStore.from_documents(chunks, self.embeddings)
        else:
            self.vector_store.add_documents(chunks)
        self.vector_store.save()

        return {
            "files": [path.name for path in saved_paths],
            "chunks_added": len(chunks),
            "total_chunks": self.indexed_chunks,
            "preview": chunk_preview(chunks),
            "warnings": [],
        }

    def clear_index(self) -> None:
        if FAISS_DIR.exists():
            shutil.rmtree(FAISS_DIR)
        ensure_data_dirs()
        self.vector_store = None

    def ask(self, question: str) -> dict:
        nodes = SelfRagNodes(
            settings=self.settings,
            llm=self._get_llm(),
            vector_store=self.vector_store,
            reranker_factory=self._get_reranker if self.vector_store is not None else None,
        )
        graph = build_self_rag_graph(nodes, self.settings.max_self_rag_iterations)
        result = graph.invoke(
            {
                "question": question,
                "query_type": "research",
                "retrieve_decision": "retrieve",
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

        citations = _build_citations(result.get("filtered_chunks", []))
        return {
            "answer": result.get("final_answer") or result.get("draft_answer", ""),
            "query_type": result.get("query_type"),
            "retrieve_decision": result.get("retrieve_decision"),
            "groundedness": result.get("groundedness"),
            "utility": result.get("utility"),
            "iterations": result.get("iteration", 0),
            "trace": result.get("trace", []),
            "citations": citations,
            "supporting_chunks": _build_supporting_chunks(result.get("filtered_chunks", [])),
        }

    def _get_llm(self):
        if self.llm is None:
            self.llm = build_llm(self.settings)
        return self.llm

    def _get_reranker(self) -> CrossEncoderReranker:
        if self.reranker is None:
            self.reranker = CrossEncoderReranker(self.settings)
        return self.reranker


def _build_citations(chunks: list[Document]) -> list[dict[str, str]]:
    citations: list[dict[str, str]] = []
    seen: set[str] = set()
    for chunk in chunks:
        source = str(chunk.metadata.get("source_file", "unknown"))
        page = str(chunk.metadata.get("page", "?"))
        key = f"{source}:{page}"
        if key in seen:
            continue
        seen.add(key)
        citations.append({"source": source, "page": page})
    return citations


def _build_supporting_chunks(chunks: list[Document]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for idx, chunk in enumerate(chunks, start=1):
        text = " ".join(chunk.page_content.split())
        items.append(
            {
                "label": (
                    f"Context {idx}: {chunk.metadata.get('source_file', 'unknown')} "
                    f"(page {chunk.metadata.get('page', '?')})"
                ),
                "text": text[:700] + ("..." if len(text) > 700 else ""),
            }
        )
    return items


def _safe_pdf_filename(filename: str) -> str:
    name = Path(filename).name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem).strip("._")
    if not stem:
        stem = "uploaded"
    return f"{stem}.pdf"


def _unique_upload_path(path: Path) -> Path:
    if not path.exists():
        return path

    counter = 2
    while True:
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1
