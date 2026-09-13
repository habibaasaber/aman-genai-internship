"""
app/nodes.py
All LangGraph node functions for the Talent Pipeline agent.

Nodes:
  1. ingestion_prescreening  — extract candidate info + knockout check
  2. skill_analysis          — compare skills, compute score
  3. interview               — set disposition = "Interview"
  4. phone_screen            — set disposition = "Phone Screen"
  5. rejection               — set disposition = "Rejected"
  6. generate_ats_payload    — build final ATS JSON
"""

import re
from typing import Any

from app.state import TalentPipelineState, SkillAnalysis
from app.llm import get_llm_extractor


# Singleton: build the extractor once when the module is loaded.
_extractor = get_llm_extractor()


# ---------------------------------------------------------------------------
# Node 1 — Ingestion & Pre-Screening
# ---------------------------------------------------------------------------

def ingestion_prescreening(state: TalentPipelineState) -> dict[str, Any]:
    """
    Extract structured candidate information from the resume using the LLM
    (or fallback extractor), then perform knockout requirement checks.

    Knockout check: each knockout requirement is checked as a keyword
    (case-insensitive) inside the resume text.

    Returns updated state fields.
    """
    resume_text = state["resume_text"]
    knockout_requirements: list[str] = state.get("knockout_requirements", [])

    # --- Step 1: LLM / fallback extraction ---
    extraction = _extractor(resume_text)

    name = extraction.name
    email = extraction.email
    phone = extraction.phone
    years_experience = extraction.years_experience
    skills: list[str] = extraction.skills or []

    # --- Step 2: Knockout checks ---
    # Each knockout requirement is matched as a keyword in the resume text.
    # If ANY requirement is absent → Fail.
    pre_screening_status = "Pass"
    rejection_reason: str | None = None

    for requirement in knockout_requirements:
        if not re.search(re.escape(requirement), resume_text, re.IGNORECASE):
            pre_screening_status = "Fail"
            rejection_reason = (
                f"Candidate did not meet the non-negotiable requirement: '{requirement}'"
            )
            break  # fail fast on first missing requirement

    # Also check minimum years of experience if detectable
    # (The job_description encodes this via knockout_requirements list)
    # e.g. "3 years experience" in knockout_requirements will be caught above.

    return {
        "name": name,
        "email": email,
        "phone": phone,
        "years_experience": years_experience,
        "skills": skills,
        "pre_screening_status": pre_screening_status,
        "rejection_reason": rejection_reason,
    }


# ---------------------------------------------------------------------------
# Node 2 — Technical Skill Analysis
# ---------------------------------------------------------------------------

def skill_analysis(state: TalentPipelineState) -> dict[str, Any]:
    """
    Compare candidate skills (from state) against required job skills.
    Comparison is case-insensitive.

    score = len(matched) / len(required_skills) * 100  (rounded to int)
    """
    required_skills: list[str] = state.get("required_skills", [])
    candidate_skills: list[str] = state.get("skills", [])

    # Normalise to lowercase for comparison
    required_lower = [s.lower() for s in required_skills]
    candidate_lower = [s.lower() for s in candidate_skills]

    matched = [
        req for req in required_skills
        if req.lower() in candidate_lower
    ]
    missing = [
        req for req in required_skills
        if req.lower() not in candidate_lower
    ]
    additional = [
        cand for cand in candidate_skills
        if cand.lower() not in required_lower
    ]

    score = round(len(matched) / len(required_skills) * 100) if required_skills else 0

    analysis: SkillAnalysis = {
        "matched": matched,
        "missing": missing,
        "additional": additional,
        "score": score,
    }

    return {"skill_analysis": analysis}


# ---------------------------------------------------------------------------
# Node 3 — Interview
# ---------------------------------------------------------------------------

def interview(state: TalentPipelineState) -> dict[str, Any]:
    """Candidate qualifies for a full interview."""
    return {
        "final_disposition": "Interview",
        "rejection_reason": None,
    }


# ---------------------------------------------------------------------------
# Node 4 — Phone Screen
# ---------------------------------------------------------------------------

def phone_screen(state: TalentPipelineState) -> dict[str, Any]:
    """Candidate qualifies for a phone screen."""
    return {
        "final_disposition": "Phone Screen",
        "rejection_reason": None,
    }


# ---------------------------------------------------------------------------
# Node 5 — Rejection
# ---------------------------------------------------------------------------

def rejection(state: TalentPipelineState) -> dict[str, Any]:
    """Candidate is rejected. Provide a generic but clear reason if not already set."""
    existing_reason = state.get("rejection_reason")

    if not existing_reason:
        # Came from skill_analysis route — build a reason from scores
        analysis = state.get("skill_analysis")
        if analysis is not None:
            score = analysis.get("score", 0)
            missing = analysis.get("missing", [])
            missing_str = ", ".join(missing) if missing else "several required skills"
            existing_reason = (
                f"Skill match score of {score}% is below the minimum threshold (50%). "
                f"Missing required skills: {missing_str}."
            )
        else:
            existing_reason = "Candidate did not meet the minimum requirements."

    return {
        "final_disposition": "Rejected",
        "rejection_reason": existing_reason,
    }


# ---------------------------------------------------------------------------
# Node 6 — Generate ATS Payload
# ---------------------------------------------------------------------------

def generate_ats_payload(state: TalentPipelineState) -> dict[str, Any]:
    """
    Build the final structured ATS JSON payload from the accumulated state.
    This is the last node on every path.
    """
    payload: dict = {
        "name": state.get("name"),
        "email": state.get("email"),
        "phone": state.get("phone"),
        "years_experience": state.get("years_experience"),
        "skills": state.get("skills", []),
        "skill_analysis": state.get("skill_analysis"),
        "final_disposition": state.get("final_disposition"),
    }

    # Include rejection_reason only when rejected
    if state.get("final_disposition") == "Rejected":
        payload["rejection_reason"] = state.get("rejection_reason")

    return {"ats_payload": payload}
