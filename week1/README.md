# Prompt Stress-Test Suite

This repository contains a production-ready evaluation suite for testing different Large Language Models (LLMs) and Prompting Strategies against real-world business tasks.

## 🏗️ Architecture

```mermaid
graph TD
    A[run.py (Entry Point)] -->|1. Loads Data| B(inputs.json)
    A -->|2. Reads Config| C(config.yaml)
    A -->|3. Triggers| D(Evaluator)
    
    D -->|4. Loads Prompts| E[prompts/ dir]
    D -->|5. Checks| F(JSON Cache)
    
    F -- Cache Miss --> G[BaseLLMClient]
    G --> H[OpenAIClient]
    G --> I[GeminiClient]
    
    D -->|6. Calculates| J(Metrics Module)
    J --> K[Token Counter & Cost Estimator]
    
    D -->|7. Returns DataFrame| L[ReportGenerator]
    
    L --> M[raw/results.csv]
    L --> N[processed/scores.xlsx]
    L --> O[charts/ *.png]
    L --> P[report.md]
```

## 🚀 Setup & Execution

1. **Install Requirements:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables:**
   Create a `.env` file in the root directory and add your API keys:
   ```env
   OPENAI_API_KEY=sk-...
   GEMINI_API_KEY=AI...
   ```

3. **Run the Evaluation Suite:**
   ```bash
   # Make sure you are in the week1 directory and run:
   set PYTHONPATH=.
   python src/run.py
   ```

4. **View the Results:**
   - Raw data: `outputs/raw/results_raw.csv`
   - Processed metrics: `outputs/processed/scores.xlsx`
   - Visualizations: `outputs/charts/`
   - Markdown Report: `report.md`
   - Interactive Demo: `notebooks/Week1_Demo.ipynb`

## 🧠 Engineering Principles Applied

- **SOLID & Clean Architecture:** Abstracted base classes for models (`BaseLLMClient`).
- **Decoupling:** Prompts are template files, business logic is separate from presentation.
- **Cost Efficiency:** Local caching prevents redundant API calls to save credits and time.
- **Config-Driven:** `config.yaml` dictates execution behavior and model parameters without code changes.
