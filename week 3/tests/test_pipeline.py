"""
tests/test_pipeline.py
Unit and integration tests for the Talent Pipeline agent.

Tests use the FALLBACK extractor (no API key required) so they run
fully offline.  We achieve this by patching the module-level _extractor
in app.nodes before the graph is built.

Test coverage:
  1. Pre-screening PASS
  2. Pre-screening FAIL (knockout)
  3. Score > 80  → Interview
  4. 50 <= Score <= 80 → Phone Screen
  5. Score < 50  → Rejected
  6. End-to-end Interview flow (strong candidate)
  7. End-to-end Rejection flow (knockout candidate)
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from app.llm import CandidateExtraction, _fallback_extractor
from app.routing import route_after_prescreening, route_after_skill_analysis
from app.state import TalentPipelineState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data"
JD_PATH = DATA_DIR / "job_description.json"


def load_job_data() -> dict:
    with open(JD_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_resume(name: str) -> str:
    return (DATA_DIR / "resumes" / name).read_text(encoding="utf-8")


def make_base_state(**overrides) -> TalentPipelineState:
    """Return a minimal valid TalentPipelineState for testing."""
    jd = load_job_data()
    state: TalentPipelineState = {
        "resume_text": "dummy resume text Bachelor degree",
        "job_description": jd["description"],
        "required_skills": jd["required_skills"],
        "knockout_requirements": jd["knockout_requirements"],
        "name": None,
        "email": None,
        "phone": None,
        "years_experience": None,
        "skills": [],
        "pre_screening_status": "Pass",
        "skill_analysis": None,
        "final_disposition": None,
        "rejection_reason": None,
        "ats_payload": None,
    }
    state.update(overrides)
    return state


def build_pipeline_with_fallback():
    """Build the compiled graph, patching the extractor to the fallback."""
    with patch("app.nodes._extractor", side_effect=_fallback_extractor):
        # Re-import graph module to pick up the patch in nodes
        from app.graph import build_graph
        return build_graph()


# ---------------------------------------------------------------------------
# Unit tests — routing functions
# ---------------------------------------------------------------------------

class TestRouting:

    def test_prescreening_pass_routes_to_skill_analysis(self):
        state = make_base_state(pre_screening_status="Pass")
        assert route_after_prescreening(state) == "skill_analysis"

    def test_prescreening_fail_routes_to_rejection(self):
        state = make_base_state(pre_screening_status="Fail")
        assert route_after_prescreening(state) == "rejection"

    def test_score_above_80_routes_to_interview(self):
        state = make_base_state(skill_analysis={"matched": [], "missing": [], "additional": [], "score": 100})
        assert route_after_skill_analysis(state) == "interview"

    def test_score_exactly_81_routes_to_interview(self):
        state = make_base_state(skill_analysis={"matched": [], "missing": [], "additional": [], "score": 81})
        assert route_after_skill_analysis(state) == "interview"

    def test_score_80_routes_to_phone_screen(self):
        state = make_base_state(skill_analysis={"matched": [], "missing": [], "additional": [], "score": 80})
        assert route_after_skill_analysis(state) == "phone_screen"

    def test_score_50_routes_to_phone_screen(self):
        state = make_base_state(skill_analysis={"matched": [], "missing": [], "additional": [], "score": 50})
        assert route_after_skill_analysis(state) == "phone_screen"

    def test_score_below_50_routes_to_rejection(self):
        state = make_base_state(skill_analysis={"matched": [], "missing": [], "additional": [], "score": 49})
        assert route_after_skill_analysis(state) == "rejection"

    def test_score_0_routes_to_rejection(self):
        state = make_base_state(skill_analysis={"matched": [], "missing": [], "additional": [], "score": 0})
        assert route_after_skill_analysis(state) == "rejection"

    def test_no_skill_analysis_routes_to_rejection(self):
        state = make_base_state(skill_analysis=None)
        assert route_after_skill_analysis(state) == "rejection"


# ---------------------------------------------------------------------------
# Unit tests — node logic
# ---------------------------------------------------------------------------

class TestSkillAnalysisNode:
    """Test the skill_analysis node directly."""

    def test_all_skills_matched(self):
        from app.nodes import skill_analysis
        state = make_base_state(
            skills=["Python", "API Integration", "SQL", "Docker", "React"],
        )
        result = skill_analysis(state)
        analysis = result["skill_analysis"]
        assert analysis["score"] == 100
        assert len(analysis["matched"]) == 5
        assert len(analysis["missing"]) == 0

    def test_some_skills_matched(self):
        from app.nodes import skill_analysis
        state = make_base_state(
            skills=["Python", "SQL", "Docker"],  # 3 out of 5
        )
        result = skill_analysis(state)
        analysis = result["skill_analysis"]
        assert analysis["score"] == 60
        assert "Python" in analysis["matched"]
        assert "React" in analysis["missing"]

    def test_no_skills_matched(self):
        from app.nodes import skill_analysis
        state = make_base_state(skills=["HTML", "CSS", "WordPress"])
        result = skill_analysis(state)
        assert result["skill_analysis"]["score"] == 0
        assert len(result["skill_analysis"]["matched"]) == 0

    def test_additional_skills_captured(self):
        from app.nodes import skill_analysis
        state = make_base_state(
            skills=["Python", "API Integration", "SQL", "Docker", "React", "Kubernetes"],
        )
        result = skill_analysis(state)
        assert "Kubernetes" in result["skill_analysis"]["additional"]


class TestIngestionPrescreeningNode:
    """Test the pre-screening node with the fallback extractor."""

    def test_prescreening_pass(self):
        from app.nodes import ingestion_prescreening
        with patch("app.nodes._extractor", side_effect=_fallback_extractor):
            state = make_base_state(
                resume_text=(
                    "Jane Smith\njane@email.com\n"
                    "Bachelor of Science in Computer Science\n"
                    "6 years of experience\nSkills: Python, Docker"
                )
            )
            result = ingestion_prescreening(state)
        assert result["pre_screening_status"] == "Pass"

    def test_prescreening_fail_missing_knockout(self):
        from app.nodes import ingestion_prescreening
        with patch("app.nodes._extractor", side_effect=_fallback_extractor):
            state = make_base_state(
                resume_text=(
                    "Chris Morgan\nchris@email.com\n"
                    "Self-taught, no formal degree.\n"
                    "5 years of experience\nSkills: Python, Docker"
                )
            )
            result = ingestion_prescreening(state)
        assert result["pre_screening_status"] == "Fail"
        assert result["rejection_reason"] is not None


# ---------------------------------------------------------------------------
# End-to-end integration tests (using fallback extractor)
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """Full pipeline runs using resume files and fallback extractor."""

    def _run(self, resume_filename: str) -> dict:
        jd = load_job_data()
        resume_text = load_resume(resume_filename)

        initial_state: TalentPipelineState = {
            "resume_text": resume_text,
            "job_description": jd["description"],
            "required_skills": jd["required_skills"],
            "knockout_requirements": jd["knockout_requirements"],
            "name": None,
            "email": None,
            "phone": None,
            "years_experience": None,
            "skills": [],
            "pre_screening_status": "Pass",
            "skill_analysis": None,
            "final_disposition": None,
            "rejection_reason": None,
            "ats_payload": None,
        }

        from app.graph import build_graph
        with patch("app.nodes._extractor", side_effect=_fallback_extractor):
            graph = build_graph()
            return graph.invoke(initial_state)

    def test_strong_candidate_interview(self):
        result = self._run("strong.txt")
        assert result["pre_screening_status"] == "Pass"
        assert result["final_disposition"] == "Interview"
        assert result["skill_analysis"]["score"] > 80
        assert result["ats_payload"] is not None

    def test_knockout_candidate_rejected(self):
        result = self._run("knockout_fail.txt")
        # The knockout_fail.txt has "No Bachelor degree awarded" — should fail
        assert result["final_disposition"] == "Rejected"
        assert result["rejection_reason"] is not None

    def test_ats_payload_structure(self):
        """Verify the ATS payload has the required fields."""
        result = self._run("strong.txt")
        payload = result["ats_payload"]
        required_keys = {"name", "email", "phone", "years_experience",
                         "skills", "skill_analysis", "final_disposition"}
        assert required_keys.issubset(set(payload.keys()))
        analysis = payload["skill_analysis"]
        assert "matched" in analysis
        assert "missing" in analysis
        assert "additional" in analysis
        assert "score" in analysis

    def test_rejection_includes_reason(self):
        result = self._run("knockout_fail.txt")
        payload = result["ats_payload"]
        assert "rejection_reason" in payload
        assert payload["rejection_reason"] is not None

    def test_interview_no_rejection_reason_in_payload(self):
        result = self._run("strong.txt")
        payload = result["ats_payload"]
        # rejection_reason key should NOT be present for Interview
        assert "rejection_reason" not in payload
