"""
app/graph.py
Builds and compiles the LangGraph StateGraph for the Talent Pipeline.

Graph topology:
  START
    ↓
  ingestion_prescreening
    ↓ (conditional)
    ├── rejection ──────────────────────────────┐
    └── skill_analysis                          │
          ↓ (conditional)                       │
          ├── interview    → generate_ats_payload│
          ├── phone_screen → generate_ats_payload│
          └── rejection    ──────────────────────┘
                                    ↓
                                  END
"""

from langgraph.graph import StateGraph, START, END

from app.state import TalentPipelineState
from app.nodes import (
    ingestion_prescreening,
    skill_analysis,
    interview,
    phone_screen,
    rejection,
    generate_ats_payload,
)
from app.routing import route_after_prescreening, route_after_skill_analysis


def build_graph() -> StateGraph:
    """Construct and compile the talent pipeline graph."""

    graph = StateGraph(TalentPipelineState)

    # ---- Register nodes ----
    graph.add_node("ingestion_prescreening", ingestion_prescreening)
    graph.add_node("skill_analysis", skill_analysis)
    graph.add_node("interview", interview)
    graph.add_node("phone_screen", phone_screen)
    graph.add_node("rejection", rejection)
    graph.add_node("generate_ats_payload", generate_ats_payload)

    # ---- Entry point ----
    graph.add_edge(START, "ingestion_prescreening")

    # ---- Conditional edge: after pre-screening ----
    graph.add_conditional_edges(
        "ingestion_prescreening",
        route_after_prescreening,
        {
            "rejection": "rejection",
            "skill_analysis": "skill_analysis",
        },
    )

    # ---- Conditional edge: after skill analysis ----
    graph.add_conditional_edges(
        "skill_analysis",
        route_after_skill_analysis,
        {
            "interview": "interview",
            "phone_screen": "phone_screen",
            "rejection": "rejection",
        },
    )

    # ---- Disposition nodes → ATS payload ----
    graph.add_edge("interview", "generate_ats_payload")
    graph.add_edge("phone_screen", "generate_ats_payload")
    graph.add_edge("rejection", "generate_ats_payload")

    # ---- ATS payload → END ----
    graph.add_edge("generate_ats_payload", END)

    return graph.compile()


def print_graph_structure() -> None:
    """Print a text summary of the compiled graph structure."""
    compiled = build_graph()
    print("\n=== LangGraph — Talent Pipeline Structure ===")
    try:
        print(compiled.get_graph().draw_ascii())
    except ImportError:
        # grandalf is optional — print a manual representation instead
        print(
            "  START\n"
            "    ↓\n"
            "  ingestion_prescreening\n"
            "    ↓ (conditional)\n"
            "    ├── rejection ──────────────────────────────┐\n"
            "    └── skill_analysis                          │\n"
            "          ↓ (conditional)                       │\n"
            "          ├── interview    → generate_ats_payload│\n"
            "          ├── phone_screen → generate_ats_payload│\n"
            "          └── rejection    ──────────────────────┘\n"
            "                                    ↓\n"
            "                                  END\n"
        )
    print("=============================================\n")
