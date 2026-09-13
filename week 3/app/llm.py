"""
app/llm.py
LLM provider factory.

Provides two extractors:
  1. get_llm_extractor()  → resume text  → CandidateExtraction
  2. get_jd_extractor()   → JD text      → JobRequirements

Supported providers (set via .env):
  LLM_PROVIDER=gemini   + GOOGLE_API_KEY=...
  LLM_PROVIDER=openai   + OPENAI_API_KEY=...

If no API key is configured, a regex/heuristic fallback is used automatically.
"""

import os
import re
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class CandidateExtraction(BaseModel):
    """
    Structured candidate information extracted from a resume.
    Works on any resume format — structured, unstructured, or story-like.
    """
    name: Optional[str] = Field(
        None,
        description="Full name of the candidate"
    )
    email: Optional[str] = Field(
        None,
        description="Email address"
    )
    phone: Optional[str] = Field(
        None,
        description="Phone number"
    )
    years_experience: Optional[float] = Field(
        None,
        description=(
            "Total years of professional work experience as a number. "
            "Only extract if explicitly stated. Do not infer or calculate."
        )
    )
    skills: list[str] = Field(
        default_factory=list,
        description=(
            "List of technical and professional skills mentioned anywhere in the resume. "
            "Include skills found inside paragraphs or natural language descriptions. "
            "Normalize capitalization (e.g. 'python' → 'Python')."
        )
    )


class JobRequirements(BaseModel):
    """
    Structured requirements extracted from a job description.
    Supports both structured and natural-language JD text.
    """
    required_skills: list[str] = Field(
        default_factory=list,
        description=(
            "List of technical or professional skills the job requires. "
            "Extract from any part of the JD text."
        )
    )
    knockout_requirements: list[str] = Field(
        default_factory=list,
        description=(
            "Non-negotiable / must-have requirements such as degree type, "
            "certifications, or clearances. Use short keyword phrases "
            "e.g. 'Bachelor', 'PMP', 'Security Clearance'."
        )
    )
    min_years_required: Optional[float] = Field(
        None,
        description=(
            "Minimum years of experience required, as a number. "
            "Only set if explicitly mentioned."
        )
    )


# ---------------------------------------------------------------------------
# Resume extractor factory
# ---------------------------------------------------------------------------

_RESUME_SYSTEM_PROMPT = """\
You are a professional resume parser. Your task is to extract structured \
information from the resume provided below.

IMPORTANT RULES:
- Extract information ONLY from what is explicitly written in the resume.
- Do NOT hallucinate, guess, or infer information that is not present.
- If a field is not mentioned, return null or an empty list.
- Extract skills from ALL parts of the resume: bullet points, paragraphs, \
  summaries, project descriptions, and natural language sentences.
- Normalize skill names (e.g. "python" → "Python", "sql" → "SQL").
- For years_experience: only extract if the resume explicitly states a number \
  of years. Do not calculate from dates.
- The resume may be structured (sections, bullets) or unstructured (paragraphs, \
  story-like text). Handle both equally well.

RESUME TEXT:
"""

def get_llm_extractor():
    """
    Return a callable:  resume_text (str) → CandidateExtraction

    Priority:
      1. Gemini  (LLM_PROVIDER=gemini  + GOOGLE_API_KEY)
      2. OpenAI  (LLM_PROVIDER=openai  + OPENAI_API_KEY)
      3. Regex fallback (no API key needed)
    """
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()

    if provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if api_key:
            return _build_gemini_extractor(api_key, CandidateExtraction, _RESUME_SYSTEM_PROMPT)
        else:
            print("[llm] GOOGLE_API_KEY not set → using fallback extractor")

    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            return _build_openai_extractor(api_key, CandidateExtraction, _RESUME_SYSTEM_PROMPT)
        else:
            print("[llm] OPENAI_API_KEY not set → using fallback extractor")

    else:
        print(f"[llm] Unknown LLM_PROVIDER '{provider}' → using fallback extractor")

    return _fallback_extractor


# ---------------------------------------------------------------------------
# Job Description extractor factory
# ---------------------------------------------------------------------------

_JD_SYSTEM_PROMPT = """\
You are a job description parser. Extract structured requirements from the \
job description below.

IMPORTANT RULES:
- required_skills: list every technical/professional skill mentioned.
- knockout_requirements: list non-negotiable requirements (degree, certification, \
  clearance). Use SHORT keyword phrases that will be matched against a resume, \
  e.g. "Bachelor", "PMP", "Security Clearance". Do NOT include experience years here.
- min_years_required: only if a minimum number of years is explicitly stated.
- Do NOT add requirements that are not in the text.

JOB DESCRIPTION:
"""

def get_jd_extractor():
    """
    Return a callable:  jd_text (str) → JobRequirements

    Uses the same provider as the resume extractor.
    Falls back to a simple regex parser if no API key is available.
    """
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()

    if provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if api_key:
            return _build_gemini_extractor(api_key, JobRequirements, _JD_SYSTEM_PROMPT)
        else:
            print("[llm] GOOGLE_API_KEY not set → using fallback JD extractor")

    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            return _build_openai_extractor(api_key, JobRequirements, _JD_SYSTEM_PROMPT)
        else:
            print("[llm] OPENAI_API_KEY not set → using fallback JD extractor")

    return _fallback_jd_extractor


# ---------------------------------------------------------------------------
# Generic Gemini builder (works for any Pydantic schema)
# ---------------------------------------------------------------------------

def _build_gemini_extractor(api_key: str, schema, system_prompt: str):
    """Build a Gemini-backed structured extractor for any Pydantic schema."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        google_api_key=api_key,
        temperature=0,
    )
    structured_llm = llm.with_structured_output(schema)

    schema_name = schema.__name__

    def extractor(text: str):
        prompt = system_prompt + text
        result = structured_llm.invoke(prompt)
        return result

    print(f"[llm] Using Gemini (gemini-flash-latest) for {schema_name} extraction")
    return extractor


# ---------------------------------------------------------------------------
# Generic OpenAI builder
# ---------------------------------------------------------------------------

def _build_openai_extractor(api_key: str, schema, system_prompt: str):
    """Build an OpenAI-backed structured extractor for any Pydantic schema."""
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=api_key,
        temperature=0,
    )
    structured_llm = llm.with_structured_output(schema)

    schema_name = schema.__name__

    def extractor(text: str):
        prompt = system_prompt + text
        result = structured_llm.invoke(prompt)
        return result

    print(f"[llm] Using OpenAI (gpt-4o-mini) for {schema_name} extraction")
    return extractor


# ---------------------------------------------------------------------------
# Fallback resume extractor (no API — pure regex / heuristics)
# ---------------------------------------------------------------------------

def _fallback_extractor(resume_text: str) -> CandidateExtraction:
    """
    Simple regex-based resume parser.
    Used when no LLM API key is configured.
    No network calls — fully local.
    Works best on structured resumes; handles unstructured text partially.
    """
    print("[llm] [FALLBACK] Using regex extractor — no LLM API called")

    # --- Email ---
    email_match = re.search(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", resume_text)
    email = email_match.group(0) if email_match else None

    # --- Phone ---
    phone_match = re.search(r"(\+?\d[\d\s\-().]{7,}\d)", resume_text)
    phone = phone_match.group(0).strip() if phone_match else None

    # --- Name: first non-empty line that looks like a personal name ---
    name = None
    for line in resume_text.splitlines():
        line = line.strip()
        if line and not re.search(r"[@\d|]", line) and len(line.split()) in (2, 3):
            name = line
            break

    # --- Years of experience ---
    years_experience = None
    exp_patterns = [
        r"(\d+(?:\.\d+)?)\+?\s*years?\s+(?:of\s+)?(?:professional\s+)?experience",
        r"experience[:\s]+(\d+(?:\.\d+)?)\+?\s*years?",
        r"(\d+(?:\.\d+)?)\s*years?\s+of\s+(?:backend|frontend|software|full.?stack|web)",
    ]
    for pat in exp_patterns:
        m = re.search(pat, resume_text, re.IGNORECASE)
        if m:
            years_experience = float(m.group(1))
            break

    # --- Skills: prefer a "Skills" section, else scan full text ---
    skills: list[str] = []
    skill_section_match = re.search(
        r"(?:skills?|technical skills?|competencies)[:\s]*(.*?)(?:\n\n|\Z)",
        resume_text,
        re.IGNORECASE | re.DOTALL,
    )
    if skill_section_match:
        raw = skill_section_match.group(1)
        tokens = re.split(r"[,|\n•\-]+", raw)
        skills = [t.strip() for t in tokens if t.strip() and len(t.strip()) > 1]
    else:
        # Scan full text for known tech keywords
        known = [
            "Python", "JavaScript", "TypeScript", "React", "Node.js", "SQL",
            "PostgreSQL", "MySQL", "MongoDB", "Docker", "Kubernetes", "AWS",
            "GCP", "Azure", "FastAPI", "Django", "Flask", "REST", "GraphQL",
            "Git", "Linux", "Java", "C++", "Go", "Rust", "Redis", "Kafka",
            "Terraform", "CI/CD", "API Integration", "Machine Learning",
        ]
        skills = [k for k in known if re.search(rf"\b{re.escape(k)}\b", resume_text, re.IGNORECASE)]

    return CandidateExtraction(
        name=name,
        email=email,
        phone=phone,
        years_experience=years_experience,
        skills=skills,
    )


# ---------------------------------------------------------------------------
# Fallback JD extractor (no API — pure regex / heuristics)
# ---------------------------------------------------------------------------

def _fallback_jd_extractor(jd_text: str) -> JobRequirements:
    """
    Simple heuristic job description parser.
    Used when no LLM API key is configured.
    """
    print("[llm] [FALLBACK] Using regex JD extractor — no LLM API called")

    known_skills = [
        "Python", "JavaScript", "TypeScript", "React", "Node.js", "SQL",
        "PostgreSQL", "MySQL", "MongoDB", "Docker", "Kubernetes", "AWS",
        "GCP", "Azure", "FastAPI", "Django", "Flask", "REST", "REST APIs",
        "GraphQL", "Git", "Linux", "Java", "C++", "Go", "Rust", "Redis",
        "Kafka", "Terraform", "CI/CD", "API Integration", "Machine Learning",
    ]
    required_skills = [
        k for k in known_skills
        if re.search(rf"\b{re.escape(k)}\b", jd_text, re.IGNORECASE)
    ]

    # Knockout: look for degree / certification phrases
    knockout = []
    if re.search(r"bachelor", jd_text, re.IGNORECASE):
        knockout.append("Bachelor")
    if re.search(r"master", jd_text, re.IGNORECASE):
        knockout.append("Master")

    # Min years
    min_years = None
    m = re.search(r"(\d+)\+?\s*years?\s+(?:of\s+)?experience", jd_text, re.IGNORECASE)
    if m:
        min_years = float(m.group(1))

    return JobRequirements(
        required_skills=required_skills,
        knockout_requirements=knockout,
        min_years_required=min_years,
    )
