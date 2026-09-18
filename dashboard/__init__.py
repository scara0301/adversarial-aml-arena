"""Streamlit explainer for the Adversarial Fraud & AML Detection Arena.

Reads the artifacts Phases 1-2 already wrote to reports/ -- never retrains or
re-evolves anything, except on the "Try the Adversary Yourself" page, which
scores a genome live against the actual trained detector.

Run with:
    uv run streamlit run dashboard/app.py
"""
