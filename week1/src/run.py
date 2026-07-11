import json
import os
from dotenv import load_dotenv

from src.utils.logger import get_logger
from src.evaluation.evaluator import Evaluator
from src.reporting.report_generator import ReportGenerator

logger = get_logger(__name__)

def main():
    logger.info("Loading environment variables from .env...")
    load_dotenv()
    
    logger.info("Loading inputs data...")
    inputs_path = os.path.join("data", "inputs.json")
    with open(inputs_path, "r", encoding="utf-8") as f:
        inputs = json.load(f)
        
    tasks = ["task1", "task2", "task3"]
    strategies = ["zero_shot", "few_shot", "chain_of_thought"]
    
    logger.info("Initializing Evaluator...")
    evaluator = Evaluator(config_path=os.path.join("config", "config.yaml"))
    
    logger.info("Starting Evaluation Pipeline...")
    results_df = evaluator.run_evaluation(tasks, strategies, inputs)
    
    logger.info("Generating Reports and Exports...")
    reporter = ReportGenerator(results_df)
    reporter.run_all()
    
    logger.info("Stress-Test Suite Execution Complete!")

if __name__ == "__main__":
    main()
