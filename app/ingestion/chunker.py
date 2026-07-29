from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pdfplumber
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.config import Settings


def load_pdfs(paths: list[Path]) -> list[Document]:
    documents: list[Document] = []
    for path in paths:
        documents.extend(_load_pdf_pages(path))
    return documents


def _load_pdf_pages(path: Path) -> list[Document]:
    pages = _load_with_pdfplumber(path)
    if pages:
        return pages
    return _load_with_pypdf(path)


def _load_with_pdfplumber(path: Path) -> list[Document]:
    documents: list[Document] = []
    with pdfplumber.open(path) as pdf:
        for page_index, page in enumerate(pdf.pages):
            text = page.extract_text(layout=True, x_tolerance=1, y_tolerance=3) or ""
            text = _clean_extracted_text(text)
            if not text:
                continue
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source_file": path.name,
                        "page": page_index,
                        "extractor": "pdfplumber",
                    },
                )
            )
    return documents


def _load_with_pypdf(path: Path) -> list[Document]:
    documents: list[Document] = []
    reader = PdfReader(str(path))
    for page_index, page in enumerate(reader.pages):
        text = _clean_extracted_text(page.extract_text() or "")
        if not text:
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source_file": path.name,
                    "page": page_index,
                    "extractor": "pypdf",
                },
            )
        )
    return documents


def _clean_extracted_text(text: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def _fallback_splitter(settings: Settings) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def _split_text_units(text: str, settings: Settings) -> list[str]:
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n+", text) if item.strip()]
    units: list[str] = []
    sentence_pattern = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")

    for paragraph in paragraphs:
        if len(paragraph) <= settings.semantic_min_chunk_size:
            units.append(paragraph)
            continue

        sentences = [item.strip() for item in sentence_pattern.split(paragraph) if item.strip()]
        buffer: list[str] = []
        buffer_len = 0
        for sentence in sentences:
            next_len = buffer_len + len(sentence) + (1 if buffer else 0)
            if buffer and next_len > settings.semantic_min_chunk_size:
                units.append(" ".join(buffer))
                buffer = [sentence]
                buffer_len = len(sentence)
            else:
                buffer.append(sentence)
                buffer_len = next_len
        if buffer:
            units.append(" ".join(buffer))

    return units or [text.strip()]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _split_oversized_chunk(text: str, metadata: dict, settings: Settings) -> list[Document]:
    splitter = _fallback_splitter(settings)
    return splitter.split_documents([Document(page_content=text, metadata=metadata)])


def _semantic_chunk_page(
    document: Document, settings: Settings, embeddings: Embeddings
) -> list[Document]:
    units = _split_text_units(document.page_content, settings)
    units = [unit for unit in units if unit.strip()]
    if not units:
        return []
    if len(units) <= 1:
        return _split_oversized_chunk(document.page_content, dict(document.metadata), settings)

    vectors = embeddings.embed_documents(units)
    chunks: list[Document] = []
    current_units: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current_units, current_len
        if not current_units:
            return
        text = "\n\n".join(current_units).strip()
        metadata = dict(document.metadata)
        metadata["chunking_strategy"] = "semantic"
        if len(text) > settings.chunk_size:
            chunks.extend(_split_oversized_chunk(text, metadata, settings))
        else:
            chunks.append(Document(page_content=text, metadata=metadata))
        current_units = []
        current_len = 0

    for idx, unit in enumerate(units):
        unit_len = len(unit)
        should_flush = False
        if current_units:
            would_exceed = current_len + unit_len + 2 > settings.chunk_size
            semantic_break = (
                _cosine_similarity(vectors[idx - 1], vectors[idx])
                < settings.semantic_similarity_threshold
                and current_len >= settings.semantic_min_chunk_size
            )
            should_flush = would_exceed or semantic_break
        if should_flush:
            flush()
        current_units.append(unit)
        current_len += unit_len + 2

    flush()
    return chunks


def chunk_documents(
    documents: list[Document],
    settings: Settings,
    embeddings: Embeddings | None = None,
) -> list[Document]:
    if settings.semantic_chunking and embeddings is not None:
        chunks: list[Document] = []
        for document in documents:
            chunks.extend(_semantic_chunk_page(document, settings, embeddings))
    else:
        splitter = _fallback_splitter(settings)
        chunks = splitter.split_documents(documents)

    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx
        chunk.metadata.setdefault("chunking_strategy", "recursive")
    return chunks


def ingest_pdf_paths(
    paths: list[Path],
    settings: Settings,
    embeddings: Embeddings | None = None,
) -> list[Document]:
    raw_docs = load_pdfs(paths)
    return chunk_documents(raw_docs, settings, embeddings)


def chunk_preview(chunks: list[Document], limit: int = 3) -> list[dict[str, Any]]:
    preview: list[dict[str, Any]] = []
    for chunk in chunks[:limit]:
        preview.append(
            {
                "source": chunk.metadata.get("source_file", "unknown"),
                "page": chunk.metadata.get("page", "?"),
                "strategy": chunk.metadata.get("chunking_strategy", "unknown"),
                "text": chunk.page_content[:280] + ("..." if len(chunk.page_content) > 280 else ""),
            }
        )
    return preview
