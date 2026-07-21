"""
app.py
======
Chainlit application for the "Ask AMAN HR" RAG chatbot.

Run with:
    chainlit run app.py

Flow
----
1. ``on_chat_start``      → Welcome banner + ChatSettings panel with pipeline
                            selector.  Both pipelines are instantiated and
                            cached in the session.
2. ``on_settings_update`` → Keep the selected pipeline in sync with the UI.
3. ``on_message``         → Route the query to the active pipeline, stream
                            status steps, display the answer, and show
                            retrieved source documents.

Pipelines available
-------------------
- **Naive RAG**    : Fixed-size chunking + pure semantic retrieval.
- **Advanced RAG** : Smart chunking + hybrid BM25+dense retrieval +
                     BGE reranking + Langfuse tracing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import chainlit as cl
from chainlit.input_widget import Select

# Ensure week2/ is on sys.path so imports resolve regardless of where
# Chainlit is launched from.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipelines.advanced_pipeline import AdvancedPipeline  # noqa: E402
from pipelines.naive_pipeline import NaivePipeline, PipelineResult  # noqa: E402
from config import settings  # noqa: E402
from utils import get_logger  # noqa: E402

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Pipeline selector options
# ---------------------------------------------------------------------------

_PIPELINE_ITEMS = {
    "🔵 Naive RAG (Baseline)": "naive",
    "🟢 Advanced RAG (Hybrid + Reranking)": "advanced",
}
_DEFAULT_PIPELINE = "naive"

# Reverse lookup: value → display label.
_PIPELINE_LABELS = {v: k for k, v in _PIPELINE_ITEMS.items()}

# Session keys.
_SESSION_PIPELINE_KEY = "pipeline_key"
_SESSION_NAIVE = "naive_pipeline"
_SESSION_ADVANCED = "advanced_pipeline"


# ---------------------------------------------------------------------------
# Welcome message
# ---------------------------------------------------------------------------

_WELCOME_MESSAGE = """\
## 🏢 مرحباً بك في مساعد AMAN للموارد البشرية
## 🏢 Welcome to the AMAN HR Assistant

Ask me anything about the **AMAN Internship Guide** — working hours, dress \
code, tracks, evaluations, graduation requirements — in **English or Arabic**.

اسألني عن أي شيء في **دليل التدريب في AMAN** — ساعات العمل، قواعد اللباس، \
المسارات، التقييمات، ومتطلبات التخرج — **بالإنجليزية أو العربية**.

---

⚙️ Use the **Settings panel** to switch between:
- 🔵 **Naive RAG** — Fixed-size chunks · Dense-only retrieval · Baseline
- 🟢 **Advanced RAG** — Smart chunks · Hybrid BM25+Dense · BGE Reranking · Langfuse Tracing

> **Tip**: Try the same question on both pipelines to compare quality!
"""

# ---------------------------------------------------------------------------
# Chat lifecycle handlers
# ---------------------------------------------------------------------------


@cl.on_chat_start
async def on_chat_start() -> None:
    """
    Initialise the session: send welcome message, render settings panel,
    and instantiate both pipelines (triggers auto-ingest if needed).
    """
    await cl.Message(content=_WELCOME_MESSAGE).send()

    # Render pipeline selector in the settings panel.
    await cl.ChatSettings(
        [
            Select(
                id=_SESSION_PIPELINE_KEY,
                label="🔄 Pipeline",
                items=_PIPELINE_ITEMS,
                initial_value=_DEFAULT_PIPELINE,
            ),
        ]
    ).send()

    # Seed session defaults.
    cl.user_session.set(_SESSION_PIPELINE_KEY, _DEFAULT_PIPELINE)

    # Instantiate pipelines during chat start so the first user message
    # doesn't pay the model-loading cost.  A loading message keeps the
    # user informed during the (potentially slow) first-time ingestion.
    async with cl.Step(
        name="🔧 Initializing pipelines…",
        show_input=False,
    ) as init_step:
        init_step.output = "Loading embedding model and connecting to Qdrant…"

        try:
            naive = await cl.make_async(NaivePipeline)(auto_ingest=True)
            cl.user_session.set(_SESSION_NAIVE, naive)
            init_step.output = "✅ Naive pipeline ready."
        except Exception as exc:
            init_step.output = f"❌ Naive pipeline failed: {exc}"
            log.exception("naive_pipeline_init_failed", error=str(exc))
            return

        try:
            advanced = await cl.make_async(AdvancedPipeline)(auto_ingest=True)
            cl.user_session.set(_SESSION_ADVANCED, advanced)
            init_step.output = "✅ Both pipelines ready. Start chatting!"
        except Exception as exc:
            init_step.output = f"❌ Advanced pipeline failed: {exc}"
            log.exception("advanced_pipeline_init_failed", error=str(exc))
            return

    log.info("chat_session_started")


@cl.on_settings_update
async def on_settings_update(settings: dict) -> None:
    """
    Sync the selected pipeline key when the user changes the Settings panel.

    Parameters
    ----------
    settings:
        Dict of widget ID → selected value from Chainlit.
    """
    pipeline_key = settings.get(_SESSION_PIPELINE_KEY, _DEFAULT_PIPELINE)
    cl.user_session.set(_SESSION_PIPELINE_KEY, pipeline_key)

    label = _PIPELINE_LABELS.get(pipeline_key, pipeline_key)
    log.info("pipeline_switched", pipeline=pipeline_key)

    await cl.Message(
        content=f"✅ Switched to **{label}**. Ask your question!",
        author="System",
    ).send()


# ---------------------------------------------------------------------------
# Message handler
# ---------------------------------------------------------------------------


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """
    Handle an incoming user message.

    Steps
    -----
    1. Retrieve the active pipeline from the session.
    2. Run the pipeline inside a spinner step.
    3. Display the generated answer.
    4. Show retrieved source documents in collapsible steps.

    Parameters
    ----------
    message:
        The Chainlit message object containing the user's query.
    """
    query = message.content.strip()
    if not query:
        await cl.Message(content="⚠️ Please enter a question.").send()
        return

    pipeline_key: str = cl.user_session.get(_SESSION_PIPELINE_KEY, _DEFAULT_PIPELINE)
    pipeline_label = _PIPELINE_LABELS.get(pipeline_key, pipeline_key)

    # Select the active pipeline instance.
    if pipeline_key == "naive":
        pipeline: NaivePipeline | AdvancedPipeline = cl.user_session.get(_SESSION_NAIVE)
    else:
        pipeline = cl.user_session.get(_SESSION_ADVANCED)

    if pipeline is None:
        await cl.Message(
            content=(
                "❌ Pipeline is not initialised. Please refresh the page "
                "to restart the session."
            )
        ).send()
        return

    log.info(
        "on_message_received",
        pipeline=pipeline_key,
        query_preview=query[:80],
    )

    # Run pipeline inside a spinner step so the user sees progress.
    async with cl.Step(
        name=f"🔍 Running {pipeline_label}…",
        show_input=False,
    ) as run_step:
        run_step.output = (
            f"**Pipeline**: {pipeline_label}\n"
            f"**Query**: {query[:100]}"
        )

        try:
            result: PipelineResult = await cl.make_async(pipeline.run)(query)
        except Exception as exc:
            await cl.Message(
                content=(
                    f"❌ An error occurred while processing your question:\n\n"
                    f"```\n{exc}\n```"
                )
            ).send()
            log.exception("pipeline_run_error", pipeline=pipeline_key, error=str(exc))
            return

        run_step.output = (
            f"✅ **{pipeline_label}** answered in {result.latency_ms:.0f} ms\n"
            f"📄 Sources used: {len(result.source_documents)}"
        )

    # --- Display the main answer ---
    pipeline_badge = "🔵" if pipeline_key == "naive" else "🟢"
    await cl.Message(
        content=(
            f"{pipeline_badge} **{pipeline_label}**\n\n"
            f"---\n\n"
            f"{result.answer}"
        ),
    ).send()

    # --- Display source documents in collapsible steps ---
    if result.source_documents:
        async with cl.Step(
            name=f"📚 Retrieved Sources ({len(result.source_documents)} documents)",
            show_input=False,
        ) as sources_step:
            source_parts: list[str] = []

            for i, doc in enumerate(result.source_documents, start=1):
                meta = doc.metadata
                page = meta.get("page", "?")
                section = meta.get("section", "")
                lang = meta.get("language", "?")
                chunk_id = meta.get("chunk_id", "?")

                # Show the most relevant score available.
                score_key = (
                    "rerank_score" if "rerank_score" in meta
                    else "rrf_score" if "rrf_score" in meta
                    else "retrieval_score"
                )
                score = meta.get(score_key)
                score_str = f"{score:.4f}" if score is not None else "N/A"

                header = (
                    f"**[{i}]** Page {page}"
                    + (f" · Section: _{section}_" if section else "")
                    + f" · Lang: `{lang}` · Chunk ID: `{chunk_id}`"
                    + f" · Score ({score_key}): `{score_str}`"
                )

                # Truncate long chunk previews to keep the UI readable.
                preview = doc.page_content[:400]
                if len(doc.page_content) > 400:
                    preview += "…"

                source_parts.append(f"{header}\n\n> {preview}")

            sources_step.output = "\n\n---\n\n".join(source_parts)

    # --- Latency footer ---
    await cl.Message(
        content=(
            f"⏱️ *Latency: {result.latency_ms:.0f} ms* "
            + (
                f"· 🔗 [View Langfuse Trace]({settings.langfuse_base_url})"
                if pipeline_key == "advanced"
                else ""
            )
        ),
        author="System",
    ).send()
