"""
app/chainlit_app.py
Chainlit web UI for the Automated Talent Pipeline Agent.

Run with:
    chainlit run app/chainlit_app.py

TWO user inputs only:
  1. CV file upload  (PDF / DOCX / TXT)
  2. Job Description (pasted text)

Chainlit responsibilities:
  - Collect the two inputs
  - Show progress messages
  - Call the existing LangGraph pipeline
  - Display the results

Business logic lives entirely in graph.py / nodes.py / routing.py.
"""

# ── Path fix ─────────────────────────────────────────────────────────────────
# Chainlit loads the file directly, which may not include the project root in
# sys.path. We add it explicitly so `from app.X import Y` always works.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent   # week 3/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Event Loop & Task Patch for Python 3.14 (fixes AnyIO NoEventLoopError & TypeError) ─
import asyncio

try:
    import sniffio
    import sniffio._impl

    _orig_sniffio_current = sniffio._impl.current_async_library

    def _patched_sniffio_current() -> str:
        try:
            return _orig_sniffio_current()
        except sniffio.AsyncLibraryNotFoundError:
            try:
                asyncio.get_running_loop()
                return "asyncio"
            except RuntimeError:
                raise

    sniffio._impl.current_async_library = _patched_sniffio_current
    sniffio.current_async_library = _patched_sniffio_current
except Exception:
    pass

try:
    import anyio._core._eventloop
    _orig_current_async_lib = anyio._core._eventloop.current_async_library

    def _patched_current_async_library() -> str | None:
        res = _orig_current_async_lib()
        if res is None:
            try:
                asyncio.get_running_loop()
                return "asyncio"
            except RuntimeError:
                pass
        return res

    anyio._core._eventloop.current_async_library = _patched_current_async_library
except Exception:
    pass

try:
    import anyio._backends._asyncio
    _orig_current_task = asyncio.current_task
    _dummy_tasks = {}

    def _patched_current_task(loop=None):
        task = _orig_current_task(loop)
        if task is not None:
            return task
        try:
            if loop is None:
                loop = asyncio.get_running_loop()
            if loop not in _dummy_tasks or _dummy_tasks[loop].done():
                async def _dummy():
                    await asyncio.sleep(3600)
                _dummy_tasks[loop] = loop.create_task(_dummy())
            return _dummy_tasks[loop]
        except RuntimeError:
            return None

    asyncio.current_task = _patched_current_task
    anyio._backends._asyncio.current_task = _patched_current_task
except Exception:
    pass
# ─────────────────────────────────────────────────────────────────────────────

import json
import os
import tempfile

import chainlit as cl
from dotenv import load_dotenv

load_dotenv()

from app.document_loader import load_document
from app.graph import build_graph
from app.llm import get_jd_extractor
from app.state import TalentPipelineState


# ---------------------------------------------------------------------------
# Chat start — guided two-step flow
# ---------------------------------------------------------------------------

@cl.on_chat_start
async def on_chat_start():
    """
    Main entry point.
    Step 1 → ask for CV file.
    Step 2 → ask for Job Description.
    Step 3 → run pipeline and display results.
    """
    try:
        # ── Welcome ──────────────────────────────────────────────────────────────
        await cl.Message(
            content=(
                "## 🤖 Automated Talent Pipeline Agent\n\n"
                "I will evaluate a candidate's CV against a job description "
                "using a LangGraph AI pipeline.\n\n"
                "---"
            )
        ).send()

        # ── Step 1: Ask for CV file ───────────────────────────────────────────────
        try:
            cv_files = await cl.AskFileMessage(
                content=(
                    "**Step 1 of 2 — Upload CV**\n\n"
                    "Please upload the candidate's resume.\n"
                    "Supported formats: **PDF, DOCX, TXT**\n\n"
                    "_The CV can be structured, semi-structured, or story-like text._"
                ),
                accept=["application/pdf",
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        "text/plain",
                        ".pdf", ".docx", ".txt"],
                max_size_mb=20,
                timeout=3600,
            ).send()
        except (TimeoutError, asyncio.TimeoutError):
            await cl.Message(
                content="⏱️ **Session timed out waiting for CV file upload.**\n\nPlease refresh the page (F5) to start a new evaluation."
            ).send()
            return

        if not cv_files:
            await cl.Message(content="⚠️ No file received. Please refresh the page (F5) to try again.").send()
            return

        cv_file = cv_files[0]

        # Extract text from the uploaded file
        await cl.Message(content="📄 Reading resume...").send()

        # Chainlit 2.x gives us cv_file.path (a temp file path)
        file_path = cv_file.path

        # Ensure the extension is preserved for the loader
        original_name: str = cv_file.name
        suffix = Path(original_name).suffix.lower()
        if not suffix:
            suffix = ".txt"

        # If the temp file doesn't have the right extension, copy it
        if not file_path.endswith(suffix):
            with open(file_path, "rb") as src:
                data = src.read()
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(data)
            tmp.close()
            file_path = tmp.name

        try:
            resume_text = load_document(file_path)
        except (ValueError, RuntimeError) as exc:
            await cl.Message(
                content=f"❌ **Could not read the file:** {exc}\n\nPlease check the file and try again."
            ).send()
            return

        char_count = len(resume_text)
        preview = resume_text[:200].replace("\n", " ").strip()
        await cl.Message(
            content=(
                f"✅ **CV loaded** — `{original_name}` ({char_count} characters)\n\n"
                f"> {preview}..."
            )
        ).send()

        # ── Step 2: Ask for Job Description ──────────────────────────────────────
        try:
            jd_response = await cl.AskUserMessage(
                content=(
                    "**Step 2 of 2 — Job Description**\n\n"
                    "Please paste the complete Job Description below.\n\n"
                    "_It can be structured or natural language — "
                    "for example: 'We are looking for a Backend Engineer with Python, SQL, "
                    "and Docker. Candidates must have a Bachelor's degree and 3+ years of experience.'_"
                ),
                timeout=3600,
            ).send()
        except (TimeoutError, asyncio.TimeoutError):
            await cl.Message(
                content="⏱️ **Session timed out waiting for Job Description.**\n\nPlease refresh the page (F5) to start a new evaluation."
            ).send()
            return

        if not jd_response or not jd_response.get("output", "").strip():
            await cl.Message(
                content="⚠️ No job description provided. Please refresh the page (F5) to try again."
            ).send()
            return

        jd_text = jd_response["output"].strip()

        # ── Step 3: Run the pipeline ──────────────────────────────────────────────
        await _run_pipeline(resume_text, jd_text)

    except (TimeoutError, asyncio.TimeoutError):
        await cl.Message(
            content="⏱️ **Request timed out.**\n\nPlease refresh the page (F5) to try again."
        ).send()
    except Exception as exc:
        await cl.Message(
            content=f"❌ **An unexpected error occurred:** `{exc}`\n\nPlease refresh the page (F5) to try again."
        ).send()


# ---------------------------------------------------------------------------
# Core pipeline runner
# ---------------------------------------------------------------------------

async def _run_pipeline(resume_text: str, jd_text: str) -> None:
    """
    Orchestrate:
      1. Extract job requirements from JD text (LLM)
      2. Build LangGraph initial state
      3. Invoke the graph
      4. Display results

    All business logic (routing, scoring, disposition) is inside the graph.
    """

    # ── Extract job requirements ──────────────────────────────────────────────
    await cl.Message(content="🤖 Extracting job requirements from Job Description...").send()

    try:
        jd_extractor = get_jd_extractor()
        job_req = jd_extractor(jd_text)
    except Exception as exc:
        await cl.Message(
            content=(
                f"❌ **Failed to extract job requirements.**\n\n"
                f"Error: `{exc}`\n\n"
                "Please check your API key in the `.env` file."
            )
        ).send()
        return

    required_skills = job_req.required_skills
    knockout_requirements = job_req.knockout_requirements

    if not required_skills:
        await cl.Message(
            content=(
                "⚠️ **No required skills detected in the Job Description.**\n\n"
                "Please make sure the JD mentions specific skills "
                "(e.g. Python, SQL, Docker, React)."
            )
        ).send()
        return

    await cl.Message(
        content=(
            "📋 **Job requirements extracted:**\n"
            f"- **Required skills:** {', '.join(required_skills)}\n"
            f"- **Knockout requirements:** {', '.join(knockout_requirements) or '_None_'}\n"
            f"- **Min experience:** "
            f"{job_req.min_years_required if job_req.min_years_required else '_Not specified_'} years"
        )
    ).send()

    # ── Build initial LangGraph state ─────────────────────────────────────────
    initial_state: TalentPipelineState = {
        "resume_text": resume_text,
        "job_description": jd_text,
        "required_skills": required_skills,
        "knockout_requirements": knockout_requirements,
        # Fields populated by nodes during execution
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

    # ── Run LangGraph ─────────────────────────────────────────────────────────
    await cl.Message(content="🤖 Extracting candidate information from CV...").send()
    await cl.Message(content="🔍 Running pre-screening checks...").send()

    try:
        graph = build_graph()
        result = graph.invoke(initial_state)
    except Exception as exc:
        await cl.Message(
            content=(
                f"❌ **Pipeline error:**\n\n`{exc}`\n\n"
                "Please check the API key and resume content."
            )
        ).send()
        return

    await cl.Message(content="📊 Analyzing skill match...").send()
    await cl.Message(content="🎯 Generating final recommendation...").send()
    await cl.Message(content="✅ **Analysis complete!**\n\n---").send()

    # ── Display results ───────────────────────────────────────────────────────
    await _display_results(result)


# ---------------------------------------------------------------------------
# Result display
# ---------------------------------------------------------------------------

async def _display_results(result: dict) -> None:
    """Format and send each result section as a separate Chainlit message."""

    payload       = result.get("ats_payload") or {}
    analysis      = result.get("skill_analysis") or {}
    pre_status    = result.get("pre_screening_status", "Unknown")
    disposition   = result.get("final_disposition")
    reject_reason = result.get("rejection_reason")

    # ── 1. Candidate Information ──────────────────────────────────────────────
    yoe = payload.get("years_experience")
    skills_list = payload.get("skills") or []

    await cl.Message(
        content=(
            "## 👤 Candidate Information\n\n"
            f"| Field | Value |\n"
            f"|---|---|\n"
            f"| **Name** | {payload.get('name') or '_Not found_'} |\n"
            f"| **Email** | {payload.get('email') or '_Not found_'} |\n"
            f"| **Phone** | {payload.get('phone') or '_Not found_'} |\n"
            f"| **Years of Experience** | {yoe if yoe is not None else '_Not found_'} |\n\n"
            f"**Extracted Skills:**\n"
            + (
                "\n".join(f"- {s}" for s in skills_list)
                if skills_list else "_No skills extracted_"
            )
        )
    ).send()

    # ── 2. Pre-Screening ──────────────────────────────────────────────────────
    if pre_status == "Pass":
        pre_badge = "✅ **PASS**"
        pre_detail = "_All non-negotiable requirements satisfied._"
    else:
        pre_badge = "❌ **FAIL**"
        pre_detail = (
            f"> **Reason:** {reject_reason}"
            if reject_reason
            else "> Candidate did not meet a non-negotiable requirement."
        )

    await cl.Message(
        content=f"## 🔍 Pre-Screening\n\n{pre_badge}\n\n{pre_detail}"
    ).send()

    # ── 3. Skill Analysis (only shown when pre-screening passed) ─────────────
    if analysis:
        score      = analysis.get("score", 0)
        matched    = analysis.get("matched", [])
        missing    = analysis.get("missing", [])
        additional = analysis.get("additional", [])

        def fmt(items):
            return "\n".join(f"- {s}" for s in items) if items else "_None_"

        await cl.Message(
            content=(
                f"## 📊 Skill Analysis\n\n"
                f"**Skill Match Score: {score}%**\n\n"
                f"**Matched Skills** ({len(matched)}):\n{fmt(matched)}\n\n"
                f"**Missing Skills** ({len(missing)}):\n{fmt(missing)}\n\n"
                f"**Additional Skills** (not required, {len(additional)}):\n{fmt(additional)}"
            )
        ).send()

    # ── 4 & 5. Final Decision + Rejection Explanation ─────────────────────────
    if disposition == "Interview":
        decision_icon = "🟢"
        decision_text = "**INTERVIEW**"
        detail = "_This candidate meets the requirements for a full interview._"
    elif disposition == "Phone Screen":
        decision_icon = "🟡"
        decision_text = "**PHONE SCREEN**"
        detail = "_This candidate partially meets the requirements and should be phone-screened._"
    else:
        decision_icon = "🔴"
        decision_text = "**REJECTED**"
        if reject_reason:
            detail = f"> {reject_reason}"
        else:
            detail = "> This candidate did not meet the minimum requirements."

    await cl.Message(
        content=(
            f"## 🎯 Final Decision\n\n"
            f"{decision_icon} {decision_text}\n\n"
            f"{detail}"
        )
    ).send()

    # ── 6. ATS Payload JSON ───────────────────────────────────────────────────
    await cl.Message(
        content=(
            "## 📋 ATS Payload\n\n"
            "```json\n"
            + json.dumps(payload, indent=2, ensure_ascii=False)
            + "\n```"
        )
    ).send()

    # ── 7. Downloadable JSON file ─────────────────────────────────────────────
    candidate_name = (payload.get("name") or "candidate").replace(" ", "_")
    filename = f"ats_{candidate_name}.json"
    json_bytes = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")

    await cl.Message(
        content="💾 **Download ATS Payload:**",
        elements=[
            cl.File(
                name=filename,
                content=json_bytes,
                mime="application/json",
            )
        ],
    ).send()

    await cl.Message(
        content=(
            "---\n_To evaluate another candidate, "
            "please refresh the page and start a new session._"
        )
    ).send()
