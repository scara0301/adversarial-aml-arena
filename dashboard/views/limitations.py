"""Honest limitations — stated here rather than discovered at demo time."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    st.title("⚠️ Limitations & What's Next")

    st.subheader("What this dashboard is (and isn't)")
    st.markdown(
        """
This is an **explainer for Phases 1-2**, built ahead of Phase 3. It reads the
static artifacts those two phases already produced
(`reports/phase{1,2}_metrics.json`) and never retrains or re-evolves
anything, except on **Try the Adversary Yourself**, which scores a genome
you choose against the real trained detector live.

There is no round-by-round arms-race curve here yet — that's Phase 3's
closed-loop orchestrator, not built. This dashboard shows round 0 only: one
detector, one evolved adversary, no retraining loop between them.
"""
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
    with st.expander("PaySim will not transfer cleanly"):
        st.write(
            "Different schema, no comparable account graph or typology labels "
            "— the relational features will not carry over. Fallback: validate "
            "only the per-transaction feature subset."
        )
    with st.expander("The adversary was evolved against one fixed detector"):
        st.write(
            "That's what Phase 2 asks for, but it makes the recall gap shown on "
            "the Phase 2 page an *upper bound* on evasion. A detector retrained "
            "on evolved strategies should be considerably harder to fool — the "
            "entire point of the Phase 3 closed loop."
        )

    st.subheader("What Phase 3 needs from here")
    st.markdown(
        """
- The **diverse strategy pool** (8 mutually-distinct evolved genomes) to fold
  back into the detector's training data.
- The **naive → evolved recall gap** as round 0 of the arms-race curve.
- `AccountGraph.overlay` and its ~5ms-per-candidate cost, which is what makes
  re-evolving the adversary once per round affordable.
"""
    )
