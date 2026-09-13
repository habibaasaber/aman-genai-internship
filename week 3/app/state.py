"""
app/state.py
Shared state definition for the Talent Pipeline LangGraph agent.
Uses TypedDict for LangGraph compatibility.
"""

from typing import TypedDict, Optional


class SkillAnalysis(TypedDict):
    """Result of comparing candidate skills to required job skills."""
    matched: list[str]
    missing: list[str]
    additional: list[str]
    score: int  # percentage: matched_required / total_required * 100


class TalentPipelineState(TypedDict):
    """
    Shared state that flows through every node in the LangGraph pipeline.
    Fields are populated progressively as the graph executes.
    """

    # --- Inputs (provided before graph execution) ---
    resume_text: str
    job_description: str
    required_skills: list[str]          # skills the job requires
    knockout_requirements: list[str]    # non-negotiable requirements (checked as keywords)

    # --- Extracted from resume (Node 1) ---
    name: Optional[str]
    email: Optional[str]
    phone: Optional[str]
    years_experience: Optional[float]
    skills: list[str]

    # --- Pre-screening result (Node 1) ---
    pre_screening_status: str           # "Pass" | "Fail"

    # --- Skill analysis result (Node 2) ---
    skill_analysis: Optional[SkillAnalysis]

    # --- Disposition (set by interview / phone_screen / rejection node) ---
    final_disposition: Optional[str]    # "Interview" | "Phone Screen" | "Rejected"
    rejection_reason: Optional[str]

    # --- Final ATS payload (last node) ---
    ats_payload: Optional[dict]
