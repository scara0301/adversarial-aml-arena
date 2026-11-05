"""Landing page: the pitch, the headline result, and the four invariants."""

from __future__ import annotations

import streamlit as st

from dashboard import data

INVARIANTS = [
    (
        "1. The adversary emits transactions, never features",
        "One `featurize()` serves both offline training and online GA evaluation. "
        "An adversary that edited feature vectors directly would be cheating and "
        "would void the entire premise.",
        "Guarded by the batch-vs-overlay equivalence test — measured max difference: **0.0**.",
    ),
    (
        "2. Money conservation is a decoder invariant, not a repaired constraint",
        "The genome encodes only relative quantities (splits, weights, fractions), "
        "never absolute amounts. The decoder always allocates exactly the goal "
        "amount, so *any* genome — including arbitrary crossover offspring — "
        "conserves money by construction.",
        "Tested over 1,500 random crossover offspring — max conservation error: "
        "**$2×10⁻¹⁰** on a $250,000 goal.",
    ),
    (
        "3. The decision threshold comes from a fixed policy, never re-tuned per round",
        "The threshold is the score quantile that spends a fixed alert budget "
        "(0.5% of transactions) on the frozen clean holdout. If it moved freely "
        "between rounds, the arms-race curve would measure threshold drift, not "
        "learning.",
        None,
    ),
    (
        "4. The frozen clean holdout never sees adversary data",
        "Carved out in Phase 1, before the adversary existed. All cross-round "
        "metrics are computed on it, so a rising recall curve would mean a real "
        "change in detection ability — not evaluation-set contamination.",
        None,
    ),
]

PHASES = [
    ("0", "Scaffold, canonical schema, config", "✅ done"),
    ("1", "Synthetic data, feature pipeline, LightGBM detector", "✅ done"),
    ("2", "GA adversary: genome, decoder, constraints, fitness", "✅ done"),
    ("3", "Closed-loop arena: retraining, arms-race curve", "✅ done"),
    ("4", "Dashboard: flows, attribution drift, decision projection", "✅ done"),
    ("5", "Real IBM AML data; transfer and feature-group checks", "✅ done"),
    ("6", "Final consolidated report", "⬜ not started"),
]


def render() -> None:
    st.title("🕵️ Adversarial Fraud & AML Detection Arena")
    st.caption("Batch A5 mini-project · Guide: Dr. B R Vatsala")

    st.markdown(
        """
A closed-loop system that pits a transaction fraud/AML detector against a
self-improving **genetic-algorithm adversary** in a continuous arms race —
instead of validating detection against one static dataset.

Most fraud-detection projects report one accuracy number against fraud
patterns already known at training time. This one asks a harder question:
**how good is the detector against an opponent that is actively trying to
evade it — and is still improving?**
"""
    )

    p1 = data.phase1_metrics()
    p2 = data.phase2_metrics()

    st.subheader("The headline result so far")
    if p1 and p2:
        h = p2["headline"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Detector recall — naive laundering", f"{h['detector_recall_naive']:.1%}")
        c2.metric(
            "Detector recall — evolved laundering",
            f"{h['detector_recall_evolved']:.1%}",
            delta=f"-{h['recall_drop']:.1%} vs naive",
            delta_color="inverse",
        )
        c3.metric("Goal delivered by the evolved strategy", f"{p2['evolved']['completion']:.1%}")
        st.info(
            "That gap is the whole point of the project: a detector that catches "
            f"~{h['detector_recall_naive']:.0%} of naive laundering catches only "
            f"~{h['detector_recall_evolved']:.0%} of the same money moved by an evolved "
            "strategy — while the strategy still delivers essentially the full amount. "
            "See **Phase 2 — The Adversary** for how it got there."
        )
        p3 = data.phase3_metrics()
        if p3:
            own = [a["champion_vs_own"] for a in p3["adversaries"]]
            st.markdown(
                f"**And with the loop closed** ({len(own)} rounds of retraining on the adversary's "
                f"own strategies): detector recall against the adversary built to beat it rises from "
                f"{own[0]['detector_recall']:.0%} to {own[-1]['detector_recall']:.0%}, and the "
                f"adversary's success rate falls from {own[0]['success_rate']:.0%} to "
                f"{own[-1]['success_rate']:.0%} — but the adversary still wins most episodes. "
                "The detector gains ground every round and never gets ahead. See "
                "**Phase 3 — The Arms Race**, including the three flaws the experiment found "
                "in itself on the way."
            )
    else:
        st.warning(
            "No results yet. Run these first:\n\n"
            "```bash\nuv run python scripts/make_synthetic.py\n"
            "uv run python scripts/train_detector.py\n"
            "uv run python scripts/evolve_adversary.py\n```"
        )

    st.subheader("The four invariants")
    st.caption(
        "Not style preferences — each is the difference between a real result "
        "and a meaningless one, and each is checked by a test."
    )
    for title, body, evidence in INVARIANTS:
        with st.expander(title):
            st.write(body)
            if evidence:
                st.success(evidence)

    st.subheader("Phase status")
    st.table(
        {
            "Phase": [p[0] for p in PHASES],
            "Scope": [p[1] for p in PHASES],
            "Status": [p[2] for p in PHASES],
        }
    )
