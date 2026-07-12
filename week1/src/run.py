import json
import sys
from pathlib import Path

from dotenv import load_dotenv

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.evaluation.evaluator import Evaluator
from src.reporting.report_generator import ReportGenerator
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    logger.info("Loading environment variables from .env...")
    load_dotenv(project_root / ".env")

    logger.info("Loading inputs data...")
    inputs_path = project_root / "data" / "inputs.json"
    with open(inputs_path, "r", encoding="utf-8") as f:
        inputs = json.load(f)

    tasks = ["task1", "task2", "task3"]
    strategies = ["zero_shot", "few_shot", "chain_of_thought"]

    logger.info("Initializing Evaluator...")
    evaluator = Evaluator(config_path=project_root / "config" / "config.yaml")

    logger.info("Starting Evaluation Pipeline...")
    results_df = evaluator.run_evaluation(tasks, strategies, inputs)

    logger.info("Generating Reports and Exports...")
    reporter = ReportGenerator(results_df, project_root=project_root)
    reporter.run_all()

    logger.info("Stress-Test Suite Execution Complete!")


if __name__ == "__main__":
    main()
