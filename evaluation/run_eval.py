from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.evaluator import run_benchmark_evaluation


def main() -> None:
    print("Running Golden Questions Benchmark Evaluation...")
    summary = run_benchmark_evaluation()
    print("\n=== Benchmark Evaluation Summary ===")
    print(f"Total Queries: {summary['total']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Citation Rate: {summary['citation_rate']:.1f}%")
    print(f"Escalation Accuracy: {summary['escalation_accuracy']:.1f}%")
    print("=====================================\n")


if __name__ == "__main__":
    main()
