"""
app/main.py
Entry point for the Automated Talent Pipeline Agent.

Runs all four sample candidates through the LangGraph pipeline and prints:
  - A routing summary  (Pre-screening status, skill score, final disposition)
  - The final ATS JSON payload

Usage:
  python -m app.main
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from app.graph import build_graph, print_graph_structure
from app.state import TalentPipelineState


# ---------------------------------------------------------------------------
# Job description & requirements (loaded from data/job_description.json)
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data"


def load_job_description() -> dict:
    jd_path = DATA_DIR / "job_description.json"
    with open(jd_path, encoding="utf-8") as f:
        return json.load(f)


def load_resume(filename: str) -> str:
    resume_path = DATA_DIR / "resumes" / filename
    return resume_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Run a single candidate
# ---------------------------------------------------------------------------

def run_candidate(
    label: str,
    resume_filename: str,
    job_data: dict,
    graph,
) -> None:
    """Run one candidate through the pipeline and print results."""
    resume_text = load_resume(resume_filename)

    initial_state: TalentPipelineState = {
        "resume_text": resume_text,
        "job_description": job_data["description"],
        "required_skills": job_data["required_skills"],
        "knockout_requirements": job_data["knockout_requirements"],
        # Fields populated by nodes — initialise to None / empty
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

    result = graph.invoke(initial_state)

    # --- Print header ---
    header = f"Candidate: {label}"
    print("\n" + "=" * 60)
    print(header)
    print("=" * len(header))

    # --- Routing summary ---
    pre_status = result.get("pre_screening_status", "Unknown")
    skill_score = (
        result["skill_analysis"]["score"]
        if result.get("skill_analysis")
        else "N/A (pre-screening failed)"
    )
    disposition = result.get("final_disposition", "Unknown")

    print(f"\nPre-screening : {pre_status}")
    print(f"Skill score   : {skill_score}")
    print(f"Final         : {disposition}")
    if result.get("rejection_reason"):
        print(f"Reason        : {result['rejection_reason']}")

    # --- ATS Payload ---
    print("\n--- ATS Payload ---")
    print(json.dumps(result.get("ats_payload"), indent=2))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n[START] Automated Talent Pipeline Agent\n")

    # Optionally display graph structure
    try:
        print_graph_structure()
    except Exception:
        pass  # ASCII drawing may fail on some environments — not critical

    job_data = load_job_description()
    graph = build_graph()

    candidates = [
        ("Strong",        "strong.txt"),
        ("Medium",        "medium.txt"),
        ("Weak",          "weak.txt"),
        ("Knockout Fail", "knockout_fail.txt"),
    ]

    for label, filename in candidates:
        try:
            run_candidate(label, filename, job_data, graph)
        except Exception as exc:
            print(f"\n[ERROR] Failed to process '{label}': {exc}")

    print("=" * 60)
    print("[DONE] All candidates processed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
