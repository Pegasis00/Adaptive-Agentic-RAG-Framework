from __future__ import annotations

from functools import lru_cache

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from app.config import Settings


@lru_cache(maxsize=2)
def get_reranker(model_name: str) -> CrossEncoder:
    return CrossEncoder(model_name)


class CrossEncoderReranker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = get_reranker(settings.reranker_model)

    def rerank(self, query: str, documents: list[Document]) -> list[Document]:
        if not documents:
            return []

        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.model.predict(pairs)
        ranked = sorted(
            zip(documents, scores, strict=True),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        top_k = self.settings.top_k_rerank
        reranked: list[Document] = []
        for doc, score in ranked[:top_k]:
            metadata = dict(doc.metadata)
            metadata["rerank_score"] = float(score)
            reranked.append(Document(page_content=doc.page_content, metadata=metadata))
        return reranked
