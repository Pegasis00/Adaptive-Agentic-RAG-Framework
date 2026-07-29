from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document


QueryType = Literal["code", "research"]
RetrieveDecision = Literal["retrieve", "noretrieve"]
RelevanceLabel = Literal["relevant", "irrelevant"]
GroundednessLabel = Literal["supported", "unsupported"]
UtilityLabel = Literal["useful", "notuseful"]


class AgentState(TypedDict):
    question: str
    query_type: QueryType
    retrieve_decision: RetrieveDecision
    retrieved_chunks: list[Document]
    reranked_chunks: list[Document]
    filtered_chunks: list[Document]
    draft_answer: str
    final_answer: str
    relevance_labels: list[str]
    groundedness: GroundednessLabel
    utility: UtilityLabel
    issues: list[str]
    iteration: int
    trace: Annotated[list[str], operator.add]
