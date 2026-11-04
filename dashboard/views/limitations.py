"""Honest limitations — stated here rather than discovered at demo time."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    st.title("⚠️ Limitations & What's Next")

    st.subheader("What this dashboard is (and isn't)")
    st.markdown(
        """
This is an **explainer for Phases 1-3**. It reads the static artifacts those
phases wrote (`reports/phase{1,2,3}_metrics.json`) and never retrains or
re-evolves anything, except on **Try the Adversary Yourself**, which scores a
genome you choose against the real round-0 detector live.

It is not the final Phase 4 dashboard: there is no live feed (everything is a
replay of a finished run), and no SHAP or decision-boundary view yet.
"""
    )

    st.subheader("Limitations of the closed loop (Phase 3)")
    with st.expander("The adversary can query the detector's exact scores"):
        st.write(
            "The GA scores every candidate through the detector's real probabilities "
            "and its real alert threshold. A real launderer learns only whether a "
            "transfer was frozen or not — a far weaker signal. So this adversary is "
            "stronger than a realistic one, which makes the detector's numbers "
            "conservative: a real attacker would find it harder to evade, not easier."
        )
    with st.expander("One run, one seed per setting"):
        st.write(
            "The arms-race curve comes from a single run. The GA is stochastic, so a "
            "difference of a few points between two rounds can be noise, and run-to-run "
            "variance has not been measured. Large effects (the detector's recall against "
            "a strategy jumping after it was shown that strategy) are well outside that "
            "noise; small round-to-round wobbles are not evidence of anything."
        )
    with st.expander("Feedback is positive-only, and capped"):
        st.write(
            "Only the adversary's laundering operations are fed back — nothing is added "
            "to teach the detector that similar-looking *benign* behaviour is fine. The "
            "cap (20% of the original positives) and the always-retrain-from-the-original-"
            "set rule limit how far that can drag the detector, and the clean-holdout "
            "curve measures what it costs; but it is a real asymmetry."
        )
    with st.expander("The background world does not react"):
        st.write(
            "Benign customers behave identically in every round. In a real bank, "
            "tightening detection changes legitimate behaviour too, and criminals "
            "adapt on the scale of months, not generations. This is a model of an arms "
            "race, not a forecast of one."
        )

    st.subheader("Stated limitations, carried over from the project docs")
    with st.expander("You cannot plot a 74-dimensional decision boundary"):
        st.write(
            "Despite the synopsis phrasing. The honest substitutes are a 2-D "
            "projection with score contours, and SHAP feature-importance drift "
            "between rounds — which shows *which* features the adversary forces "
            "the detector to stop trusting. Neither is built yet."
        )
    with st.expander("Synthetic data is not real data"):
        st.write(
            "All numbers on this dashboard are a pipeline check, not a claim "
            "about real laundering. Expect them to shift on the real IBM AML "
            "dataset (Phase 5) — likely down, given the true prevalence is "
            "~0.1% versus the 0.2% used here."
        )
    with st.expander("The synthetic laundering is never a round number"):
        st.write(
            "21.6% of benign amounts are round multiples of $100; none of the laundering "
            "is. So 'round' means 'benign' with certainty — a free shortcut the detector "
            "learned (it was the #2 feature in Phase 1). Retraining without it costs about "
            "3 points of recall and 2 of PR-AUC, so the Phase 1 numbers are roughly that "
            "much optimistic. The first closed-loop run's adversary found the shortcut in "
            "one round (a single round-number transfer; detector recall 0.000). Episode "
            "goals are now drawn to the cent, which closes the adversary's access to it — "
            "but the generator is unchanged, and fixing it is the first job before real data."
        )
    with st.expander("PaySim will not transfer cleanly"):
        st.write(
            "Different schema, no comparable account graph or typology labels "
            "— the relational features will not carry over. Fallback: validate "
            "only the per-transaction feature subset."
        )

    st.subheader("What's next")
    st.markdown(
        """
- **Phase 5** — the real IBM AML dataset, and PaySim for the per-transaction
  features only. The loader contract (`load_ledger("ibm", ...)`) is written;
  it has never been run on real data.
- **Phase 6** — the consolidated final report.
- **Not yet built**: SHAP feature-importance drift between rounds, and the 2-D
  projection with score contours.
"""
    )
