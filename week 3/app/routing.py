"""
app/routing.py
Deterministic conditional edge functions for the LangGraph pipeline.
The LLM is NOT involved in routing decisions.
"""

from app.state import TalentPipelineState


def route_after_prescreening(state: TalentPipelineState) -> str:
    """
    After Node 1 (ingestion & pre-screening):
      - "Fail"  → go to rejection node
      - "Pass"  → go to skill_analysis node
    """
    if state.get("pre_screening_status") == "Fail":
        return "rejection"
    return "skill_analysis"


def route_after_skill_analysis(state: TalentPipelineState) -> str:
    """
    After Node 2 (skill analysis) — purely score-based routing:
      score > 80  → interview
      50 ≤ score ≤ 80 → phone_screen
      score < 50  → rejection
    """
    analysis = state.get("skill_analysis")
    if analysis is None:
        return "rejection"

    score = analysis.get("score", 0)

    if score > 80:
        return "interview"
    elif score >= 50:
        return "phone_screen"
    else:
        return "rejection"
