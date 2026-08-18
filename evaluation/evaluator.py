from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

from rag.chatbot import ProductAssistant
from rag.config import ROOT_DIR

logger = logging.getLogger(__name__)
GOLDEN_CSV = ROOT_DIR / "evaluation" / "golden_questions.csv"


def run_benchmark_evaluation(csv_path: Path = GOLDEN_CSV) -> dict[str, Any]:
    """Run benchmark queries against ProductAssistant and return summary metrics."""
    if not csv_path.exists():
        return {"total": 0, "passed": 0, "failed": 0, "citation_rate": 0.0, "escalation_accuracy": 0.0, "results": []}

    assistant = ProductAssistant()
    results = []
    passed_count = 0
    total_count = 0
    citation_count = 0
    escalation_correct_count = 0

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_count += 1
            q_id = row.get("id", "")
            product = row.get("product", "")
            h_ver = row.get("hardware_version", "")
            question = row.get("question", "")
            answerable = row.get("answerable", "yes").lower() == "yes"

            try:
                result = assistant.answer(question, [], active_product=product, active_version=h_ver)
                has_citations = len(result.citations) > 0
                if has_citations:
                    citation_count += 1

                passed = False
                if answerable:
                    # An answerable question passes if it returned an answer with citations and did not escalate unexpectedly
                    passed = not result.escalated and has_citations
                else:
                    # An unanswerable question passes if it correctly escalated
                    passed = result.escalated
                    if result.escalated:
                        escalation_correct_count += 1

                if passed:
                    passed_count += 1

                results.append(
                    {
                        "id": q_id,
                        "product": product,
                        "hardware_version": h_ver,
                        "question": question,
                        "answer": result.answer,
                        "escalated": result.escalated,
                        "citations": [c.__dict__ for c in result.citations],
                        "passed": passed,
                    }
                )
            except Exception as exc:
                logger.error(f"Error evaluating Q{q_id}: {exc}")
                results.append(
                    {
                        "id": q_id,
                        "product": product,
                        "hardware_version": h_ver,
                        "question": question,
                        "answer": f"Error: {exc}",
                        "escalated": True,
                        "citations": [],
                        "passed": False,
                    }
                )

    citation_rate = (citation_count / total_count * 100.0) if total_count > 0 else 0.0
    escalation_acc = (escalation_correct_count / max(1, total_count) * 100.0)

    return {
        "total": total_count,
        "passed": passed_count,
        "failed": total_count - passed_count,
        "citation_rate": citation_rate,
        "escalation_accuracy": escalation_acc,
        "results": results,
    }
