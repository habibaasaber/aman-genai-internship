import os
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ReportGenerator:
    """Generates artifacts, charts, and markdown reports from evaluation results."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.raw_dir = "outputs/raw"
        self.processed_dir = "outputs/processed"
        self.charts_dir = "outputs/charts"
        
        for d in [self.raw_dir, self.processed_dir, self.charts_dir]:
            os.makedirs(d, exist_ok=True)
            
    def export_data(self) -> None:
        """Exports raw and processed dataframes."""
        csv_path = os.path.join(self.raw_dir, "results_raw.csv")
        self.df.to_csv(csv_path, index=False)
        logger.info(f"Exported raw CSV to {csv_path}")
        
        # In a real scenario, 'processed' might involve aggregations or cleaned responses
        excel_path = os.path.join(self.processed_dir, "scores.xlsx")
        self.df.to_excel(excel_path, index=False)
        logger.info(f"Exported processed Excel to {excel_path}")

    def generate_charts(self) -> None:
        """Generates latency and cost comparison charts."""
        # Cost Comparison
        plt.figure(figsize=(10, 6))
        cost_agg = self.df.groupby('Model')['Estimated_Cost_USD'].sum()
        cost_agg.plot(kind='bar', color=['#1f77b4', '#ff7f0e'])
        plt.title('Total Estimated Cost by Model')
        plt.ylabel('Cost (USD)')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.charts_dir, "cost_comparison.png"))
        plt.close()
        
        # Latency Comparison
        plt.figure(figsize=(10, 6))
        latency_agg = self.df.groupby('Model')['Latency_sec'].mean()
        latency_agg.plot(kind='bar', color=['#2ca02c', '#d62728'])
        plt.title('Average Latency by Model')
        plt.ylabel('Latency (Seconds)')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.charts_dir, "latency_comparison.png"))
        plt.close()
        logger.info("Exported charts to outputs/charts/")
        
    def generate_markdown_report(self) -> None:
        """Generates a summary markdown report."""
        summary = self.df.groupby('Model').agg({
            'Latency_sec': 'mean',
            'Estimated_Cost_USD': 'sum',
            'Total_Tokens': 'sum'
        }).reset_index()
        
        report_content = f"""# Prompt Stress-Test Suite Results

## Executive Summary
This report summarizes the performance of different LLMs across multiple prompting strategies for 3 specific business tasks.

### Overall Metrics
{summary.to_markdown(index=False)}

## Visualizations
### Cost Comparison
![Cost Comparison](outputs/charts/cost_comparison.png)

### Latency Comparison
![Latency Comparison](outputs/charts/latency_comparison.png)

## JSON Output Reliability (Task 1)
"""
        task1_df = self.df[self.df["Task"] == "task1"]
        if not task1_df.empty:
            json_success = task1_df.groupby(['Model', 'Strategy'])['Valid_JSON'].mean() * 100
            json_success_md = json_success.reset_index().rename(columns={'Valid_JSON': 'Success Rate (%)'}).to_markdown(index=False)
            report_content += f"Percentage of valid JSON outputs:\n\n{json_success_md}\n"
            
        with open("report.md", "w", encoding="utf-8") as f:
            f.write(report_content)
        logger.info("Generated report.md")

    def run_all(self):
        """Runs all generation steps."""
        self.export_data()
        self.generate_charts()
        self.generate_markdown_report()
