"""
app.py  –  Chainlit UI for the Prompt Stress-Test Suite
=========================================================
Run with:
    chainlit run app.py

Flow
----
1. on_chat_start  → welcome message + ChatSettings panel with three Select widgets.
2. on_settings_update → keep session values in sync whenever the user changes a widget.
3. on_message     → grab input text, call inference_service.run_inference, display result.
"""

import sys
from pathlib import Path

import chainlit as cl
from chainlit.input_widget import Select

# ---------------------------------------------------------------------------
# Ensure week1/ is on sys.path so `from src.services...` resolves correctly
# regardless of where Chainlit is launched from.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.services.inference_service import run_inference  # noqa: E402

# ---------------------------------------------------------------------------
# Widget option definitions
# items = {display_label: internal_value}
# ---------------------------------------------------------------------------

MODEL_ITEMS = {
    "GPT-4o-mini":      "gpt_4o_mini",
    "Gemini-3.5-Flash": "gemini_3.5_flash",
}

TASK_ITEMS = {
    "Task 1 – Log Triage":      "task1",
    "Task 2 – Arabic PII":      "task2",
    "Task 3 – Complaint Reply": "task3",
}

STRATEGY_ITEMS = {
    "Zero Shot":        "zero_shot",
    "Few Shot":         "few_shot",
    "Chain of Thought": "chain_of_thought",
}

# Defaults (first entry of each dict)
_DEFAULT_MODEL    = next(iter(MODEL_ITEMS.values()))
_DEFAULT_TASK     = next(iter(TASK_ITEMS.values()))
_DEFAULT_STRATEGY = next(iter(STRATEGY_ITEMS.values()))

# Reverse lookup: internal_value → display_label
_MODEL_LABELS    = {v: k for k, v in MODEL_ITEMS.items()}
_TASK_LABELS     = {v: k for k, v in TASK_ITEMS.items()}
_STRATEGY_LABELS = {v: k for k, v in STRATEGY_ITEMS.items()}


# ---------------------------------------------------------------------------
# Chat start
# ---------------------------------------------------------------------------

@cl.on_chat_start
async def on_chat_start():
    # Welcome banner
    await cl.Message(
        content=(
            "## 🧪 Welcome to the Prompt Stress-Test UI\n\n"
            "Use the **Settings panel** (⚙️ icon) to pick your model, task, and strategy.\n"
            "Then type your input text below and press **Enter** to run inference."
        )
    ).send()

    # Seed the session with defaults
    cl.user_session.set("model",    _DEFAULT_MODEL)
    cl.user_session.set("task",     _DEFAULT_TASK)
    cl.user_session.set("strategy", _DEFAULT_STRATEGY)

    # Render the settings panel
    await cl.ChatSettings(
        [
            Select(
                id="model",
                label="🤖 Model",
                items=MODEL_ITEMS,
                initial_value=_DEFAULT_MODEL,
            ),
            Select(
                id="task",
                label="📋 Task",
                items=TASK_ITEMS,
                initial_value=_DEFAULT_TASK,
            ),
            Select(
                id="strategy",
                label="💡 Prompting Strategy",
                items=STRATEGY_ITEMS,
                initial_value=_DEFAULT_STRATEGY,
            ),
        ]
    ).send()


# ---------------------------------------------------------------------------
# Settings update – keep session values in sync
# ---------------------------------------------------------------------------

@cl.on_settings_update
async def on_settings_update(settings: dict):
    cl.user_session.set("model",    settings.get("model",    _DEFAULT_MODEL))
    cl.user_session.set("task",     settings.get("task",     _DEFAULT_TASK))
    cl.user_session.set("strategy", settings.get("strategy", _DEFAULT_STRATEGY))


# ---------------------------------------------------------------------------
# On message – run inference, display result
# ---------------------------------------------------------------------------

@cl.on_message
async def on_message(message: cl.Message):
    input_text = message.content.strip()
    if not input_text:
        await cl.Message(content="⚠️ Please enter some input text.").send()
        return

    model_key = cl.user_session.get("model")
    task      = cl.user_session.get("task")
    strategy  = cl.user_session.get("strategy")

    model_label    = _MODEL_LABELS.get(model_key, model_key)
    task_label     = _TASK_LABELS.get(task, task)
    strategy_label = _STRATEGY_LABELS.get(strategy, strategy)

    # Spinner step while model is thinking
    async with cl.Step(name="⏳ Running inference…", show_input=False) as step:
        step.output = (
            f"**Model:** {model_label}  \n"
            f"**Task:** {task_label}  \n"
            f"**Strategy:** {strategy_label}"
        )

        try:
            result_dict = await cl.make_async(run_inference)(
                model_key=model_key,
                task=task,
                strategy=strategy,
                input_text=input_text,
            )
            response_text = result_dict["response"]
        except Exception as exc:
            await cl.Message(content=f"❌ Inference failed:\n\n```\n{exc}\n```").send()
            return

    # Final result card
    await cl.Message(
        content=(
            f"### ✅ Result\n\n"
            f"| Field    | Value |\n"
            f"|----------|-------|\n"
            f"| Model    | {model_label} |\n"
            f"| Task     | {task_label} |\n"
            f"| Strategy | {strategy_label} |\n\n"
            f"---\n\n"
            f"Response\n"
            f"------------------------\n"
            f"{response_text}\n\n"
            f"Execution Metrics\n"
            f"------------------------\n"
            f"Latency: {result_dict['latency_sec']:.2f} sec\n"
            f"Prompt Tokens: {result_dict['prompt_tokens']}\n"
            f"Completion Tokens: {result_dict['completion_tokens']}\n"
            f"Total Tokens: {result_dict['total_tokens']}\n"
            f"Estimated Cost: ${result_dict['estimated_cost_usd']:.6f}\n"
            f"Accuracy Score: {result_dict['accuracy_score']}\n"
            f"Valid JSON: {result_dict['valid_json'] if result_dict['valid_json'] is not None else 'N/A'}"
        )
    ).send()