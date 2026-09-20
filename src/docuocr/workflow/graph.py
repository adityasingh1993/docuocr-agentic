from __future__ import annotations

from typing import Any, Literal

from docuocr.state import DocumentState

from .nodes import WorkflowNodes


def build_graph(nodes: WorkflowNodes, *, checkpointer: Any = None) -> Any:
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(DocumentState)
    builder.add_node("ingest", nodes.ingest)
    builder.add_node("assess_quality", nodes.assess_quality)
    builder.add_node("enhance_document", nodes.enhance_document)
    builder.add_node("extraction_start", nodes.extraction_start)
    builder.add_node("layout", nodes.layout)
    builder.add_node("ocr", nodes.ocr)
    builder.add_node("controls", nodes.controls)
    builder.add_node("map_rules", nodes.map_rules)
    builder.add_node("vlm_map", nodes.vlm_map)
    builder.add_node("score", nodes.score)
    builder.add_node("recover", nodes.recover)
    builder.add_node("review", nodes.review)
    builder.add_node("assemble", nodes.assemble)

    builder.add_edge(START, "ingest")
    builder.add_edge("ingest", "assess_quality")
    builder.add_conditional_edges(
        "assess_quality",
        lambda state: _quality_route(state, nodes),
        {"enhance": "enhance_document", "extract": "extraction_start"},
    )
    builder.add_edge("enhance_document", "assess_quality")
    builder.add_edge("extraction_start", "layout")
    builder.add_edge("extraction_start", "ocr")
    builder.add_edge("extraction_start", "controls")
    builder.add_edge(["layout", "ocr", "controls"], "map_rules")
    builder.add_edge("map_rules", "vlm_map")
    builder.add_edge("vlm_map", "score")
    builder.add_conditional_edges(
        "score",
        _score_route,
        {"recover": "recover", "review": "review", "complete": "assemble"},
    )
    builder.add_edge("recover", "score")
    builder.add_edge("review", "assemble")
    builder.add_edge("assemble", END)
    return builder.compile(checkpointer=checkpointer)


def _quality_route(
    state: DocumentState, nodes: WorkflowNodes
) -> Literal["enhance", "extract"]:
    quality = float(state.get("quality", {}).get("overall", 0.0))
    attempts = int(state.get("document_attempt", 0))
    if (
        quality < nodes.settings.policy.document_quality_threshold
        and attempts < nodes.settings.policy.max_document_enhancements
    ):
        return "enhance"
    return "extract"


def _score_route(state: DocumentState) -> Literal["recover", "review", "complete"]:
    dispositions = {
        value.get("disposition") for value in state.get("decisions", {}).values()
    }
    if "retry" in dispositions:
        return "recover"
    if dispositions - {"accepted"}:
        return "review"
    return "complete"
