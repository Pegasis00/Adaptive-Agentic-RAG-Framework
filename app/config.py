from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
FAISS_DIR = DATA_DIR / "faiss_index"
MAX_UPLOAD_FILES = 20


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    groq_model: str
    embedding_model: str
    reranker_model: str
    chunk_size: int
    chunk_overlap: int
    semantic_chunking: bool
    semantic_min_chunk_size: int
    semantic_similarity_threshold: float
    hybrid_search: bool
    hybrid_dense_weight: float
    lexical_candidate_pool: int
    top_k_retrieve: int
    top_k_rerank: int
    max_self_rag_iterations: int
    max_upload_files: int

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be greater than 0.")
        if self.chunk_overlap < 0:
            raise ValueError("CHUNK_OVERLAP cannot be negative.")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
        if self.semantic_min_chunk_size <= 0:
            raise ValueError("SEMANTIC_MIN_CHUNK_SIZE must be greater than 0.")
        if not 0 <= self.semantic_similarity_threshold <= 1:
            raise ValueError("SEMANTIC_SIMILARITY_THRESHOLD must be between 0 and 1.")
        if not 0 <= self.hybrid_dense_weight <= 1:
            raise ValueError("HYBRID_DENSE_WEIGHT must be between 0 and 1.")
        if self.lexical_candidate_pool <= 0:
            raise ValueError("LEXICAL_CANDIDATE_POOL must be greater than 0.")
        if self.top_k_retrieve <= 0:
            raise ValueError("TOP_K_RETRIEVE must be greater than 0.")
        if self.top_k_rerank <= 0:
            raise ValueError("TOP_K_RERANK must be greater than 0.")
        if self.max_self_rag_iterations < 0:
            raise ValueError("MAX_SELF_RAG_ITERATIONS cannot be negative.")
        if self.max_upload_files <= 0:
            raise ValueError("MAX_UPLOAD_FILES must be greater than 0.")

    @classmethod
    def from_env(cls) -> Settings:
        def env_bool(name: str, default: str = "true") -> bool:
            return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}

        def env_int(name: str, default: str) -> int:
            value = os.getenv(name, default)
            try:
                return int(value)
            except ValueError as exc:
                raise ValueError(f"{name} must be an integer, got {value!r}.") from exc

        def env_float(name: str, default: str) -> float:
            value = os.getenv(name, default)
            try:
                return float(value)
            except ValueError as exc:
                raise ValueError(f"{name} must be a number, got {value!r}.") from exc

        return cls(
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
            groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            reranker_model=os.getenv(
                "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ),
            chunk_size=env_int("CHUNK_SIZE", "1000"),
            chunk_overlap=env_int("CHUNK_OVERLAP", "200"),
            semantic_chunking=env_bool("SEMANTIC_CHUNKING", "true"),
            semantic_min_chunk_size=env_int("SEMANTIC_MIN_CHUNK_SIZE", "350"),
            semantic_similarity_threshold=env_float("SEMANTIC_SIMILARITY_THRESHOLD", "0.58"),
            hybrid_search=env_bool("HYBRID_SEARCH", "true"),
            hybrid_dense_weight=env_float("HYBRID_DENSE_WEIGHT", "0.65"),
            lexical_candidate_pool=env_int("LEXICAL_CANDIDATE_POOL", "50"),
            top_k_retrieve=env_int("TOP_K_RETRIEVE", "30"),
            top_k_rerank=env_int("TOP_K_RERANK", "10"),
            max_self_rag_iterations=env_int("MAX_SELF_RAG_ITERATIONS", "1"),
            max_upload_files=env_int("MAX_UPLOAD_FILES", str(MAX_UPLOAD_FILES)),
        )


def ensure_data_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
