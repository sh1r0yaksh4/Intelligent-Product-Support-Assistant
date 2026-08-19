"""Run golden questions through ProductAssistant and report pass/fail verdicts.

Usage:
    python3 evaluation/run_eval.py

Requires a valid GEMINI_API_KEY in .env. This makes live API calls and is not
part of the pytest suite.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

# Provide helpful guidance if run outside the virtual environment
try:
    import chromadb
    import dotenv
    import google.genai
except ImportError:
    print("\n[Error] Dependencies not found in current Python environment.")
    print("Please activate the virtual environment first:")
    print("    source .venv/bin/activate")
    print("Or run directly using:")
    print("    .venv/bin/python3 evaluation/run_eval.py\n")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.chatbot import ProductAssistant
from rag.gemini import GeminiUnavailable

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

    # Header
    print()
    print(f"{'ID':<4} {'Product':<28} {'Question':<42} {'Expected':<10} {'Result':<12} {'Verdict'}")
    print("-" * 110)

    for row in rows:
        qid = row.get("id", "?")
        product = row.get("product", "")
        hw_version = row.get("hardware_version", "")
        question = row.get("question", "")
        answerable = row.get("answerable", "").strip().lower()

        product_label = f"{product} {hw_version}".strip()
        question_short = (question[:39] + "...") if len(question) > 42 else question

        try:
            result = assistant.answer(
                question,
                active_product=product or None,
                active_version=hw_version or None,
            )

            if answerable == "yes":
                # Expect a non-escalated answer with at least one citation
                passed = not result.escalated and len(result.citations) > 0
                actual = "answered" if not result.escalated else "escalated"
                expected_label = "answer"
            elif answerable == "no":
                # Expect an escalation (no hallucination)
                passed = result.escalated
                actual = "escalated" if result.escalated else "answered"
                expected_label = "escalate"
            else:
                passed = True
                actual = "skipped"
                expected_label = "any"

            verdict = "PASS" if passed else "FAIL"

        except GeminiUnavailable as exc:
            print(f"\n[Error] {exc}")
            print("Set GEMINI_API_KEY in .env to run evaluation.\n")
            sys.exit(1)
        except Exception as exc:
            verdict = "ERROR"
            actual = str(exc)[:20]
            expected_label = answerable if answerable in ("yes", "no") else "any"
            passed = False

        row["verdict"] = verdict
        results.append({"id": qid, "verdict": verdict, "passed": passed})

        print(f"{qid:<4} {product_label:<28} {question_short:<42} {expected_label:<10} {actual:<12} {verdict}")

        # Small delay to respect API rate limits
        time.sleep(1)

    # Summary
    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    print()
    print(f"Result: {passed_count}/{total} passed")

    # Write verdicts back to CSV
    fieldnames = list(rows[0].keys())
    with GOLDEN_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Verdicts written to {GOLDEN_CSV.name}")

    if passed_count < total:
        sys.exit(1)


if __name__ == "__main__":
    run()
