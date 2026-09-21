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
    builder.add_node("document_understand", nodes.document_understand)
    builder.add_node("map_rules", nodes.map_rules)
    builder.add_node("layout_block_map", nodes.layout_block_map)
    builder.add_node("vlm_map", nodes.vlm_map)
    builder.add_node("score", nodes.score)
    builder.add_node("recover", nodes.recover)
    builder.add_node("evidence_reverify", nodes.evidence_reverify)
    builder.add_node("review", nodes.review)
    builder.add_node("assemble", nodes.assemble)

    builder.add_edge(START, "ingest")
    builder.add_edge("ingest", "assess_quality")
    builder.add_edge("assess_quality", "extraction_start")
    builder.add_edge("extraction_start", "layout")
    builder.add_edge("extraction_start", "ocr")
    builder.add_edge("extraction_start", "controls")
    builder.add_edge(["layout", "ocr", "controls"], "document_understand")
    builder.add_conditional_edges(
        "document_understand",
        lambda state: _document_route(state, nodes),
        {"enhance": "enhance_document", "map": "map_rules"},
    )
    builder.add_conditional_edges(
        "enhance_document",
        _enhancement_route,
        {"reextract": "extraction_start", "map": "map_rules"},
    )
    builder.add_edge("map_rules", "layout_block_map")
    builder.add_edge("layout_block_map", "vlm_map")
    builder.add_edge("vlm_map", "score")
    builder.add_conditional_edges(
        "score",
        lambda state: _score_route(state, nodes),
        {
            "recover": "recover",
            "reverify": "evidence_reverify",
            "review": "review",
            "complete": "assemble",
        },
    )
    builder.add_edge("recover", "score")
    builder.add_edge("evidence_reverify", "score")
    builder.add_edge("review", "assemble")
    builder.add_edge("assemble", END)
    return builder.compile(checkpointer=checkpointer)


def _document_route(
    state: DocumentState, nodes: WorkflowNodes
) -> Literal["enhance", "map"]:
    quality_payload = state.get("quality", {})
    readiness = quality_payload.get("extraction_readiness")
    quality = float(
        readiness if readiness is not None else quality_payload.get("overall", 0.0)
    )
    attempts = int(state.get("document_attempt", 0))
    if (
        quality < nodes.settings.policy.document_quality_threshold
        and attempts < nodes.settings.policy.max_document_enhancements
    ):
        return "enhance"
    return "map"


def _enhancement_route(state: DocumentState) -> Literal["reextract", "map"]:
    if state.get("enhancement_selected", False):
        return "reextract"
    return "map"


def _score_route(
    state: DocumentState, nodes: WorkflowNodes
) -> Literal["recover", "reverify", "review", "complete"]:
    dispositions = {
        value.get("disposition") for value in state.get("decisions", {}).values()
    }
    if "retry" in dispositions:
        return "recover"
    if dispositions - {"accepted"}:
        understanding = state.get("document_understanding") or {}
        if understanding.get("schema_match") == "mismatch":
            return "review"
        settings = nodes.settings.association
        if (
            settings.evidence_reverify_enabled
            and state.get("evidence_reverification_attempt", 0)
            < settings.max_evidence_retries
        ):
            return "reverify"
        return "review"
    return "complete"
