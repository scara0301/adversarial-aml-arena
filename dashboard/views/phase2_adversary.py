"""Phase 2 walkthrough: how the GA adversary evolved, and what it found."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from arena.adversary.genome import GENES, TEMPLATES
from dashboard import data


def render() -> None:
    st.title("🧬 Phase 2 — The Adversary")
    st.caption(
        "Objectives 2-3: evolve realistic evasion strategies, guarded against "
        "degenerate shortcuts."
    )

    metrics = data.phase2_metrics()
    if metrics is None:
        st.warning("Run `uv run python scripts/evolve_adversary.py` first.")
        return

    h = metrics["headline"]
    ga = metrics["ga"]

    st.subheader("The headline result")
    c1, c2, c3 = st.columns(3)
    c1.metric("Naive laundering — detector recall", f"{h['detector_recall_naive']:.1%}")
    c2.metric(
        "Evolved laundering — detector recall",
        f"{h['detector_recall_evolved']:.1%}",
        delta=f"-{h['recall_drop']:.1%}",
        delta_color="inverse",
    )
    c3.metric(
        "Random (unevolved) — detector recall",
        f"{h['detector_recall_random']:.1%}",
        help="Confirms it's the GA search doing the work, not just randomness.",
    )
    st.caption(
        f"{ga['population']} population × {ga['generations']} generations × "
        f"{ga['episodes_per_eval']} episodes = {ga['evaluations']:,} candidate "
        f"evaluations in {metrics['runtime_minutes']} minutes."
    )

    st.divider()
    left, right = st.columns([3, 2])

    with left:
        st.subheader("Convergence")
        hist = metrics["history"]
        gens = [r["generation"] for r in hist]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=gens, y=[r["best_fitness"] for r in hist],
                name="best fitness", line=dict(color="#c53030", width=3),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=gens, y=[r["mean_fitness"] for r in hist],
                name="population mean", line=dict(color="#718096", width=2, dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=gens, y=[r["diversity"] for r in hist],
                name="population diversity", line=dict(color="#805ad5", width=1.5, dash="dot"),
                yaxis="y2",
            )
        )
        fig.update_layout(
            xaxis_title="generation",
            yaxis_title="fitness",
            yaxis_range=[0, 1.05],
            yaxis2=dict(title="diversity", overlaying="y", side="right", range=[0, 1]),
            height=380,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", y=1.15),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Watch the **mean**, not the best — elitism carries a lucky early "
            f"genome forward unchanged, so best can plateau (here: "
            f"{hist[0]['best_fitness']:.3f} → {hist[-1]['best_fitness']:.3f}) while "
            f"the population is still learning (mean: {hist[0]['mean_fitness']:.3f} → "
            f"{hist[-1]['mean_fitness']:.3f})."
        )

    with right:
        st.subheader("Fitness formula")
        st.latex(r"\text{fitness} = \text{completion} \times \text{evasion} - \gamma \cdot \text{cost}")
        st.markdown("...with a hard floor: **fitness = 0 if completion < 95%.**")
        fc = metrics["fitness_config"]
        st.dataframe(
            {"parameter": list(fc.keys()), "value": list(fc.values())},
            hide_index=True,
            use_container_width=True,
        )
        with st.expander("Why the guards?"):
            st.markdown(
                """
- **Multiplicative + completion floor** — stops "move $0, never get
  flagged" from winning.
- **Amount-weighted evasion** — stops "one giant transfer" from beating real
  structuring.
- **Episode averaging (5 per genome)** — stops the GA overfitting one lucky
  account.
- **Fitness sharing** — keeps the population diverse, for Phase 3's feedback
  loop.
"""
            )

    st.divider()
    st.subheader("Naive vs. evolved: what changed")
    best = metrics["best_genome"]
    comparison_rows = [
        {
            "gene": g.name,
            "evolved value": round(best[g.name], 3) if isinstance(best[g.name], float) else best[g.name],
            "range": f"{g.low:g}–{g.high:g}",
            "encodes": g.note,
        }
        for g in GENES
    ]
    st.dataframe(comparison_rows, hide_index=True, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        template = best.get("typology_template_name", TEMPLATES[int(best["typology_template"])])
        st.markdown(f"**Template:** `{template}`")
        st.markdown(
            f"**Completion:** {metrics['evolved']['completion']:.1%}  ·  "
            f"**Evasion:** {metrics['evolved']['evasion']:.1%}"
        )
    with c2:
        img = data.report_image("phase2_recall_comparison.png")
        if img:
            st.image(img, use_container_width=True)

    st.divider()
    st.subheader("The winning strategy, decoded")
    example = data.phase2_example_strategy()
    if example is not None and not example.empty:
        span_hours = (example["timestamp"].max() - example["timestamp"].min()).total_seconds() / 3600
        st.caption(
            f"{len(example)} transactions moving ${example['amount_paid'].sum():,.2f} "
            f"over {span_hours:.1f} hours."
        )
        st.dataframe(
            example[["timestamp", "from_account", "to_account", "amount_paid", "payment_format"]],
            hide_index=True,
            use_container_width=True,
        )

        fig3 = go.Figure(
            go.Scatter(
                x=example["timestamp"], y=example["amount_paid"], mode="markers+lines",
                marker=dict(size=10, color="#c53030"), line=dict(color="#c53030", width=1),
            )
        )
        fig3.add_hline(
            y=10000, line_dash="dash", line_color="gray",
            annotation_text="$10,000 reporting threshold",
        )
        fig3.update_layout(
            xaxis_title="time", yaxis_title="amount ($)",
            height=320, margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.info(
            "Every transaction sits **above** the $10,000 threshold — the "
            "adversary chose *not* to structure fully, because full structuring "
            "means more transactions, which trips the detector's burst features "
            "instead. See **How It Works**."
        )
    else:
        st.caption("Decoded example not available.")

    with st.expander(
        f"Diverse strategy pool ({len(metrics.get('diverse_strategies', []))} strategies for Phase 3)"
    ):
        st.caption(
            "Eight mutually-distinct genomes from the final population — feeding "
            "the detector eight copies of one winner would teach it almost nothing."
        )
        st.dataframe(
            metrics.get("diverse_strategies", []), hide_index=True, use_container_width=True
        )
