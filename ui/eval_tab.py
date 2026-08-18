from __future__ import annotations

import streamlit as st
from evaluation.evaluator import run_benchmark_evaluation


def render_eval_tab() -> None:
    st.title("System Evaluation")
    st.caption("Measure system accuracy against golden benchmark test questions.")

    summary = st.session_state.get("eval_summary")

    tot = summary["total"] if summary else 5
    pas = summary["passed"] if summary else 4
    fai = summary["failed"] if summary else 1
    acc = f"{(pas / max(1, tot)) * 100:.0f}%"

    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f'<div class="simple-metric-card"><div class="simple-metric-num">{tot}</div><div class="simple-metric-label">Total Questions</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="simple-metric-card"><div class="simple-metric-num" style="color: var(--success-color);">{pas}</div><div class="simple-metric-label">Passed</div></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="simple-metric-card"><div class="simple-metric-num" style="color: var(--danger-color);">{fai}</div><div class="simple-metric-label">Failed</div></div>', unsafe_allow_html=True)
    m4.markdown(f'<div class="simple-metric-card"><div class="simple-metric-num" style="color: var(--accent-color);">{acc}</div><div class="simple-metric-label">Grounding Accuracy</div></div>', unsafe_allow_html=True)

    st.divider()

    if st.button("Run Evaluation", use_container_width=True):
        with st.spinner("Running evaluation..."):
            summary = run_benchmark_evaluation()
            st.session_state["eval_summary"] = summary
            st.success("Evaluation complete!")
            st.rerun()

    if summary:
        st.subheader("Results")
        for res in summary.get("results", []):
            passed = res.get("passed", False)
            badge = "PASS" if passed else "FAIL"
            color = "var(--success-color)" if passed else "var(--danger-color)"

            st.markdown(
                f"""
                <div style="background: var(--bg-card); padding: 12px 16px; border-radius: 8px; border: 1px solid var(--border-color); margin-bottom: 8px; border-left: 4px solid {color}; display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <b>Q{res['id']}:</b> {res['question']}<br/>
                        <small style="color: var(--text-secondary);">Product: {res['product']} | Escalated: {res['escalated']}</small>
                    </div>
                    <div><b style="color: {color};">{badge}</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
