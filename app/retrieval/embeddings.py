from __future__ import annotations

from functools import lru_cache

from langchain_community.embeddings import HuggingFaceEmbeddings

from app.config import Settings


@lru_cache(maxsize=2)
def get_embedding_model(model_name: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_embeddings(settings: Settings) -> HuggingFaceEmbeddings:
    return get_embedding_model(settings.embedding_model)
