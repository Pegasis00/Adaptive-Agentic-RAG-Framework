from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from pathlib import Path

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.config import FAISS_DIR


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_./#-]+")


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw_token in TOKEN_PATTERN.findall(text):
        token = raw_token.lower().strip("._-/")
        if not token:
            continue
        tokens.append(token)
        normalized = _normalize_lexical_token(token)
        if normalized != token:
            tokens.append(normalized)
    return tokens


def _normalize_lexical_token(token: str) -> str:
    if len(token) > 5 and token.endswith("ships"):
        return token[:-5]
    if len(token) > 4 and token.endswith("ship"):
        return token[:-4]
    if len(token) > 5 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        return {key: 1.0 for key in scores}
    return {key: (value - low) / (high - low) for key, value in scores.items()}


def _copy_with_scores(document: Document, scores: dict[str, float]) -> Document:
    metadata = dict(document.metadata)
    metadata.update(scores)
    return Document(page_content=document.page_content, metadata=metadata)


class FaissVectorStore:
    def __init__(self, store: FAISS):
        self.store = store

    @classmethod
    def from_documents(
        cls, documents: list[Document], embeddings: Embeddings
    ) -> FaissVectorStore:
        store = FAISS.from_documents(documents, embeddings)
        return cls(store)

    @classmethod
    def load(cls, embeddings: Embeddings, path: Path | None = None) -> FaissVectorStore | None:
        index_path = path or FAISS_DIR
        if not index_path.exists() or not any(index_path.iterdir()):
            return None
        store = FAISS.load_local(
            str(index_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return cls(store)

    def save(self, path: Path | None = None) -> None:
        target = path or FAISS_DIR
        target.mkdir(parents=True, exist_ok=True)
        self.store.save_local(str(target))

    def add_documents(self, documents: list[Document]) -> None:
        self.store.add_documents(documents)

    def similarity_search(self, query: str, k: int) -> list[Document]:
        return self.store.similarity_search(query, k=k)

    def hybrid_search(
        self,
        query: str,
        k: int,
        dense_weight: float = 0.65,
        lexical_candidate_pool: int = 50,
    ) -> list[Document]:
        if k <= 0:
            return []

        documents_by_key = self._documents_by_key()
        if not documents_by_key:
            return self.similarity_search(query, k=k)

        fetch_k = min(max(k * 4, k), self.document_count)
        dense_results = self.store.similarity_search_with_score(query, k=fetch_k)
        dense_scores: dict[str, float] = {}
        for document, distance in dense_results:
            key = self._document_key(document)
            dense_scores[key] = max(dense_scores.get(key, 0.0), 1.0 / (1.0 + float(distance)))

        lexical_scores = self._lexical_scores(query, documents_by_key)
        lexical_candidate_pool = max(lexical_candidate_pool, k)
        lexical_top = dict(
            sorted(lexical_scores.items(), key=lambda item: item[1], reverse=True)[
                :lexical_candidate_pool
            ]
        )

        dense_norm = _normalize_scores(dense_scores)
        lexical_norm = _normalize_scores(lexical_top)
        keys = set(dense_norm) | set(lexical_norm)
        dense_weight = min(max(dense_weight, 0.0), 1.0)

        ranked: list[tuple[str, float]] = []
        for key in keys:
            score = (
                dense_weight * dense_norm.get(key, 0.0)
                + (1.0 - dense_weight) * lexical_norm.get(key, 0.0)
            )
            ranked.append((key, score))

        ranked.sort(key=lambda item: item[1], reverse=True)
        results: list[Document] = []
        for key, score in ranked[:k]:
            document = documents_by_key[key]
            results.append(
                _copy_with_scores(
                    document,
                    {
                        "hybrid_score": score,
                        "dense_score": dense_norm.get(key, 0.0),
                        "lexical_score": lexical_norm.get(key, 0.0),
                    },
                )
            )
        return results

    def _documents_by_key(self) -> dict[str, Document]:
        docstore = self.store.docstore
        if not isinstance(docstore, InMemoryDocstore):
            return {}
        return {self._document_key(doc): doc for doc in docstore._dict.values()}

    def _lexical_scores(
        self, query: str, documents_by_key: dict[str, Document]
    ) -> dict[str, float]:
        query_terms = _tokenize(query)
        if not query_terms:
            return {}

        query_counts = Counter(query_terms)
        doc_tokens = {
            key: _tokenize(document.page_content)
            for key, document in documents_by_key.items()
        }
        doc_lengths = {key: len(tokens) for key, tokens in doc_tokens.items()}
        avg_len = sum(doc_lengths.values()) / max(len(doc_lengths), 1)

        document_frequency: Counter[str] = Counter()
        for tokens in doc_tokens.values():
            document_frequency.update(set(tokens))

        total_docs = len(doc_tokens)
        scores: dict[str, float] = {}
        k1 = 1.5
        b = 0.75
        for key, tokens in doc_tokens.items():
            if not tokens:
                continue
            counts = Counter(tokens)
            score = 0.0
            for term, query_count in query_counts.items():
                term_frequency = counts.get(term, 0)
                if term_frequency == 0:
                    continue
                idf = math.log((total_docs - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5) + 1)
                denominator = term_frequency + k1 * (
                    1 - b + b * doc_lengths[key] / max(avg_len, 1)
                )
                score += query_count * idf * (term_frequency * (k1 + 1)) / denominator
            if score > 0:
                scores[key] = score
        return scores

    @staticmethod
    def _document_key(document: Document) -> str:
        chunk_id = document.metadata.get("chunk_id")
        source = document.metadata.get("source_file", "unknown")
        page = document.metadata.get("page", "?")
        digest = hashlib.sha1(document.page_content.encode("utf-8")).hexdigest()[:16]
        return f"{source}:{page}:{chunk_id}:{digest}"

    @property
    def document_count(self) -> int:
        docstore = self.store.docstore
        if isinstance(docstore, InMemoryDocstore):
            return len(docstore._dict)
        return 0
