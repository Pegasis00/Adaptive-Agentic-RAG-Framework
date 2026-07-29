from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.agents.nodes import SelfRagNodes
from app.agents.state import AgentState


def _route_after_retrieve_decision(state: AgentState) -> str:
    if state["retrieve_decision"] == "retrieve":
        return "retrieve"
    return "generate"


def _route_after_verification(state: AgentState, max_iterations: int = 2) -> str:
    iteration = state.get("iteration", 0)
    needs_fix = state.get("groundedness") == "unsupported" or state.get("utility") == "notuseful"
    if needs_fix and iteration < max_iterations:
        return "refine"
    return "finalize"


def build_self_rag_graph(nodes: SelfRagNodes, max_iterations: int = 2):
    graph = StateGraph(AgentState)

    graph.add_node("classify_query", nodes.classify_query)
    graph.add_node("decide_retrieve", nodes.decide_retrieve)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("rerank", nodes.rerank)
    graph.add_node("filter_relevance", nodes.filter_relevance)
    graph.add_node("generate", nodes.generate)
    graph.add_node("verify_groundedness", nodes.verify_groundedness)
    graph.add_node("verify_utility", nodes.verify_utility)
    graph.add_node("refine", nodes.refine)
    graph.add_node("finalize", nodes.finalize)

    graph.add_edge(START, "classify_query")
    graph.add_edge("classify_query", "decide_retrieve")
    graph.add_conditional_edges(
        "decide_retrieve",
        _route_after_retrieve_decision,
        {"retrieve": "retrieve", "generate": "generate"},
    )
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "filter_relevance")
    graph.add_edge("filter_relevance", "generate")
    graph.add_edge("generate", "verify_groundedness")
    graph.add_edge("verify_groundedness", "verify_utility")
    graph.add_conditional_edges(
        "verify_utility",
        lambda state: _route_after_verification(state, max_iterations),
        {"refine": "refine", "finalize": "finalize"},
    )
    graph.add_edge("refine", "verify_groundedness")
    graph.add_edge("finalize", END)

    return graph.compile()
