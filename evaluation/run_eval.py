"""Run golden benchmark questions through ProductAssistant and report pass/fail verdicts and accuracy metrics.

Usage:
    python scripts/load_demo.py
    python evaluation/run_eval.py

Requires a valid GEMINI_API_KEY in .env.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

# Ensure UTF-8 output encoding if possible
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Provide helpful guidance if run outside the virtual environment
try:
    import chromadb
    import dotenv
    import requests
except ImportError:
    print("\n[Error] Dependencies not found in current Python environment.")
    print("Please activate the virtual environment first:")
    print("    .venv\\Scripts\\activate (Windows) or source .venv/bin/activate")
    print("Or run directly using:")
    print("    .venv/bin/python3 evaluation/run_eval.py\n")
    sys.exit(1)

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from rag.chatbot import ProductAssistant
from rag.grok import GrokUnavailable

GOLDEN_CSV = Path(__file__).resolve().parent / "golden_questions.csv"


def run() -> None:
    if not GOLDEN_CSV.exists():
        print(f"Golden questions file not found: {GOLDEN_CSV}")
        sys.exit(1)

    with GOLDEN_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("No questions found in golden_questions.csv.")
        sys.exit(1)

    assistant = ProductAssistant()
    results: list[dict] = []

    print("\n" + "=" * 115)
    print(f"{'ID':<3} | {'Product':<32} | {'Question':<40} | {'Expected':<9} | {'Actual':<10} | {'Verdict'}")
    print("-" * 115)

    answered_correct = 0
    escalated_correct = 0
    total_answerable = 0
    total_unanswerable = 0

    for row in rows:
        qid = row.get("id", "?")
        product = row.get("product", "")
        hw_version = row.get("hardware_version", "")
        question = row.get("question", "")
        answerable = row.get("answerable", "").strip().lower()

        product_label = f"{product} {hw_version}".strip()
        if len(product_label) > 32:
            product_label = product_label[:29] + "..."
        question_short = (question[:37] + "...") if len(question) > 40 else question

        try:
            result = assistant.answer(
                question,
                active_product=product or None,
                active_version=hw_version or None,
            )

            if answerable == "yes":
                total_answerable += 1
                # Expect a non-escalated answer with at least one citation
                passed = not result.escalated and len(result.citations) > 0
                if passed:
                    answered_correct += 1
                actual = "answered" if not result.escalated else "escalated"
                expected_label = "answer"
            elif answerable == "no":
                total_unanswerable += 1
                # Expect an escalation (zero hallucination)
                passed = result.escalated
                if passed:
                    escalated_correct += 1
                actual = "escalated" if result.escalated else "answered"
                expected_label = "escalate"
            else:
                passed = True
                actual = "skipped"
                expected_label = "any"

            verdict = "PASS" if passed else "FAIL"

        except GrokUnavailable as exc:
            print(f"\n[Error] {exc}")
            print("Set GROK_API_KEY in .env to run evaluation.\n")
            sys.exit(1)
        except Exception as exc:
            verdict = "ERROR"
            actual = str(exc)[:15]
            expected_label = answerable if answerable in ("yes", "no") else "any"
            passed = False

        row["verdict"] = verdict
        results.append({"id": qid, "verdict": verdict, "passed": passed})

        print(f"{qid:<3} | {product_label:<32} | {question_short:<40} | {expected_label:<9} | {actual:<10} | {verdict}")

        # Slight delay for API rate limits
        time.sleep(1.0)

    # Metrics Summary
    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    accuracy_pct = (passed_count / total) * 100 if total else 0
    ans_pct = (answered_correct / total_answerable) * 100 if total_answerable else 0
    esc_pct = (escalated_correct / total_unanswerable) * 100 if total_unanswerable else 0

    print("=" * 115)
    print("\n--- Evaluation Benchmark Summary ---")
    print(f"  * Total Test Cases:          {total}")
    print(f"  * Passed:                    {passed_count}/{total} ({accuracy_pct:.1f}%)")
    print(f"  * Grounded Answer Accuracy:  {answered_correct}/{total_answerable} ({ans_pct:.1f}%)")
    print(f"  * Anti-Hallucination Guard:  {escalated_correct}/{total_unanswerable} ({esc_pct:.1f}%)\n")

    # Update CSV with verdicts
    fieldnames = list(rows[0].keys())
    with GOLDEN_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Verdicts recorded to {GOLDEN_CSV.name}")

    if passed_count < total:
        sys.exit(1)


if __name__ == "__main__":
    run()
