"""Phase 1 walkthrough: the detector's numbers, in context."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from dashboard import data


def render() -> None:
    st.title("🛡️ Phase 1 — The AML Detector")
    st.caption(
        "Objective 1: recognise fan-out, fan-in, cycle, gather-scatter and stack laundering."
    )

    metrics = data.phase1_metrics()
    if metrics is None:
        st.warning("Run `uv run python scripts/train_detector.py` first.")
        return

    h = metrics["holdout"]

    st.subheader("Why these metrics, not accuracy")
    st.markdown(
        f"""
At a **{h['positive_rate']:.3%}** positive rate, accuracy is meaningless — a model
that predicts "never laundering" scores {1 - h['positive_rate']:.1%}. Instead:

- **Recall at a fixed alert budget** — of the laundering that happened, what
  fraction gets caught while reviewing only {h['alert_budget']:.1%} of
  transactions? This mirrors how AML teams actually operate: analyst capacity
  is the binding constraint, not a probability cut-off. It's also what keeps
  recall comparable across future arms-race rounds (invariant 3).
- **PR-AUC** as a threshold-free sanity check — sensitive to the positive
  class even at this prevalence, unlike ROC-AUC.
"""
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recall @ 0.5% budget", f"{h['recall_at_budget']:.1%}", help="Primary metric")
    c2.metric("PR-AUC", f"{h['pr_auc']:.3f}", help=f"Chance level here: {h['positive_rate']:.4f}")
    c3.metric("Precision @ budget", f"{h['precision_at_budget']:.1%}")
    c4.metric("ROC-AUC", f"{h['roc_auc']:.3f}", help=h.get("roc_auc_note", ""))
    st.caption(f"⚠️ {h.get('roc_auc_note', '')}")

    st.divider()
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Recall by typology")
        st.caption(
            "The spread, not the average, is what matters — it's the weakness "
            "the adversary has an incentive to exploit."
        )
        tr = h.get("typology_recall", {})
        names = sorted(tr, key=lambda k: tr[k])
        fig = go.Figure(
            go.Bar(
                x=[tr[n] for n in names],
                y=names,
                orientation="h",
                marker_color="#2b6cb0",
                text=[f"{tr[n]:.3f}" for n in names],
                textposition="outside",
            )
        )
        fig.update_layout(
            xaxis_range=[0, 1.05],
            xaxis_title="recall at alert budget",
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, width="stretch")

    with right:
        st.subheader("Precision-recall curve")
        img = data.report_image("phase1_pr_curve.png")
        if img:
            st.image(img, width="stretch")
        else:
            st.caption("Figure not generated yet.")

    st.divider()
    st.subheader("Top features by gain")
    st.caption(
        "No single feature should dominate this chart — that was a symptom of a "
        "data leak in an earlier run, fixed before this result (see How It Works)."
    )
    top = metrics.get("top_features", {})
    items = list(top.items())[:15]
    fig2 = go.Figure(
        go.Bar(
            x=[v for _, v in items][::-1],
            y=[k for k, _ in items][::-1],
            orientation="h",
            marker_color="#2f855a",
        )
    )
    fig2.update_layout(
        xaxis_title="gain", height=420, margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig2, width="stretch")

    with st.expander("Run details"):
        st.json(
            {
                "ledger rows": metrics["rows"],
                "accounts": metrics["accounts"],
                "features": metrics["n_features"],
                "trees": metrics["n_trees"],
                "train / valid / holdout rows": metrics["split"],
                "alert budget": metrics["alert_budget"],
                "threshold (tau)": metrics["threshold"],
            }
        )
