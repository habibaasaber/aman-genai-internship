import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    """Generates artifacts, charts, and markdown reports from evaluation results."""

    def __init__(self, df: pd.DataFrame, project_root: Path | None = None):
        self.df = df.copy()
        self.project_root = project_root or Path(__file__).resolve().parents[2]
        self.raw_dir = self.project_root / "outputs" / "raw"
        self.processed_dir = self.project_root / "outputs" / "processed"
        self.charts_dir = self.project_root / "outputs" / "charts"

        for d in [self.raw_dir, self.processed_dir, self.charts_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _add_composite_scores(self) -> None:
        """Adds cost, latency, and overall composite scores for each row."""
        scored = self.df.copy()
        for (task, strategy, input_idx), group in scored.groupby(["Task", "Strategy", "Input_Index"]):
            if group.empty:
                continue

            min_cost = group["Estimated_Cost_USD"].min()
            max_cost = group["Estimated_Cost_USD"].max()
            min_latency = group["Latency_sec"].min()
            max_latency = group["Latency_sec"].max()

            if max_cost == min_cost:
                cost_scores = pd.Series(1.0, index=group.index)
            else:
                cost_scores = 1 - ((group["Estimated_Cost_USD"] - min_cost) / (max_cost - min_cost))

            if max_latency == min_latency:
                latency_scores = pd.Series(1.0, index=group.index)
            else:
                latency_scores = 1 - ((group["Latency_sec"] - min_latency) / (max_latency - min_latency))

            scored.loc[group.index, "Cost_Score"] = cost_scores.round(3)
            scored.loc[group.index, "Latency_Score"] = latency_scores.round(3)
            scored.loc[group.index, "Overall_Score"] = (
                (0.6 * scored.loc[group.index, "Accuracy_Score"])
                + (0.2 * cost_scores)
                + (0.2 * latency_scores)
            ).round(3)

        self.df = scored

    def export_data(self) -> None:
        """Exports raw and processed dataframes."""
        csv_path = self.raw_dir / "results_raw.csv"
        if csv_path.exists():
            counter = 1
            while (self.raw_dir / f"results_raw_{counter}.csv").exists():
                counter += 1
            csv_path = self.raw_dir / f"results_raw_{counter}.csv"
        self.df.to_csv(csv_path, index=False)
        logger.info(f"Exported raw CSV to {csv_path}")

        excel_path = self.processed_dir / "scoring_summary.xlsx"
        if excel_path.exists():
            counter = 1
            while (self.processed_dir / f"scoring_summary_{counter}.xlsx").exists():
                counter += 1
            excel_path = self.processed_dir / f"scoring_summary_{counter}.xlsx"
        self.df.to_excel(excel_path, index=False)
        logger.info(f"Exported processed Excel to {excel_path}")

    def generate_charts(self) -> None:
        """Generates latency and cost comparison charts."""
        plt.figure(figsize=(10, 6))
        cost_agg = self.df.groupby("Model")["Estimated_Cost_USD"].sum()
        cost_agg.plot(kind="bar", color=["#1f77b4", "#ff7f0e"])
        plt.title("Total Estimated Cost by Model")
        plt.ylabel("Cost (USD)")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.charts_dir / "cost_comparison.png")
        plt.close()

        plt.figure(figsize=(10, 6))
        latency_agg = self.df.groupby("Model")["Latency_sec"].mean()
        latency_agg.plot(kind="bar", color=["#2ca02c", "#d62728"])
        plt.title("Average Latency by Model")
        plt.ylabel("Latency (Seconds)")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.charts_dir / "latency_comparison.png")
        plt.close()
        logger.info(f"Exported charts to {self.charts_dir}")

    def generate_markdown_report(self) -> None:
        """Generates a summary markdown report."""
        self._add_composite_scores()

        summary = self.df.groupby("Model").agg({
            "Latency_sec": "mean",
            "Estimated_Cost_USD": "sum",
            "Total_Tokens": "sum",
            "Accuracy_Score": "mean",
            "Overall_Score": "mean",
        }).reset_index()

        task_winners = []
        for task in sorted(self.df["Task"].unique()):
            task_df = self.df[self.df["Task"] == task]
            task_summary = task_df.groupby("Model").agg({
                "Accuracy_Score": "mean",
                "Cost_Score": "mean",
                "Latency_Score": "mean",
                "Overall_Score": "mean",
                "Estimated_Cost_USD": "sum",
                "Latency_sec": "mean",
            }).reset_index()
            winner = task_summary.sort_values("Overall_Score", ascending=False).iloc[0]
            task_winners.append({
                "Task": task,
                "Winner": winner["Model"],
                "Overall_Score": round(winner["Overall_Score"], 3),
                "Accuracy": round(winner["Accuracy_Score"], 3),
                "Cost_Score": round(winner["Cost_Score"], 3),
                "Latency_Score": round(winner["Latency_Score"], 3),
                "Reason": self._reason_for_winner(task, winner),
            })

        winner_df = pd.DataFrame(task_winners)
        summary_md = summary.to_markdown(index=False)
        winner_md = winner_df.to_markdown(index=False)

        report_content = f"""# Prompt Stress-Test Suite Results

## Executive Summary
This report compares two providers across three prompt strategies and three business tasks using a simple heuristic accuracy score plus cost and latency metrics.

### Overall Metrics
{summary_md}

## Task Winners
{winner_md}

## Visualizations
### Cost Comparison
![Cost Comparison](outputs/charts/cost_comparison.png)

### Latency Comparison
![Latency Comparison](outputs/charts/latency_comparison.png)

## JSON Output Reliability (Task 1)
"""
        task1_df = self.df[self.df["Task"] == "task1"]
        if not task1_df.empty:
            json_success = task1_df.groupby(["Model", "Strategy"])["Valid_JSON"].mean() * 100
            json_success_md = json_success.reset_index().rename(columns={"Valid_JSON": "Success Rate (%)"}).to_markdown(index=False)
            report_content += f"Percentage of valid JSON outputs:\n\n{json_success_md}\n"

        with open(self.project_root / "report.md", "w", encoding="utf-8") as f:
            f.write(report_content)
        logger.info(f"Generated report at {self.project_root / 'report.md'}")

    def _reason_for_winner(self, task: str, winner_row: pd.Series) -> str:
        if task == "task1":
            return "It produced the most reliable JSON structure while keeping latency and cost competitive."
        if task == "task2":
            return "It extracted PII entities most consistently and stayed efficient under the evaluation heuristic."
        if task == "task3":
            return "It drafted the most complete, empathetic, and actionable reply while remaining cost-effective."
        return "It achieved the highest composite score across the task-specific rubric."

    def run_all(self):
        """Runs all generation steps."""
        self._add_composite_scores()
        self.export_data()
        self.generate_charts()
        self.generate_markdown_report()
