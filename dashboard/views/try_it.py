"""Live genome explorer.

Lets a visitor set the adversary's twelve genes with sliders, decode a real
transaction sub-ledger from them, and score it against the actual trained
detector -- through the exact `featurize()` path that scored every one of
the 20,500 candidates in the Phase 2 GA run. This is the only page in the
dashboard that touches the live model instead of a report artifact; see
dashboard/live.py.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from arena.adversary.ga import naive_genome
from arena.adversary.genome import GENES, TEMPLATES, decode_genes
from dashboard import data, live

_NAIVE = decode_genes(naive_genome())
_GROUPS = {
    "Structuring": ["n_splits", "split_skew", "threshold_hug"],
    "Timing": ["mean_delay_h", "delay_jitter", "business_hours_bias", "dormancy_h"],
    "Routing": ["n_hops", "mule_pool_size", "mule_reuse", "format_mix"],
}
_GENE_BY_NAME = {g.name: g for g in GENES}


def render() -> None:
    st.title("🧪 Try the Adversary Yourself")
    st.caption(
        "Set the twelve genes, decode a real transaction sub-ledger, and score "
        "it against the actual trained detector — the same `featurize()` path "
        "that scored every candidate in the Phase 2 GA run."
    )

    if not live.artifacts_available():
        st.warning(
            "This needs the trained detector, the synthetic ledger, and the "
            "frozen holdout on disk. Run these first:\n\n"
            "```bash\nuv run python scripts/make_synthetic.py\n"
            "uv run python scripts/train_detector.py\n"
            "uv run python scripts/evolve_adversary.py\n```"
        )
        return

    _render_presets()
    st.divider()
    _render_sliders()
    st.divider()

    if st.button("🚀 Decode & Score", type="primary", use_container_width=True):
        genome = _current_genome()
        with st.spinner("Decoding and scoring against the trained detector..."):
            outcome, sub, scores, tau = live.score_genome(genome)
        _render_result(outcome, sub, scores, tau)
    else:
        st.info("Adjust the sliders above, then click **Decode & Score**.")


def _render_presets() -> None:
    c1, c2, c3 = st.columns(3)
    if c1.button(
        "Load naive baseline", use_container_width=True,
        help="The hand-written strategy the evolved one is measured against.",
    ):
        _apply_preset(naive_genome())
    if c2.button(
        "Load GA winner", use_container_width=True,
        help="The best genome the Phase 2 evolutionary run found.",
    ):
        m = data.phase2_metrics()
        if m:
            best = m["best_genome"]
            _apply_preset(np.array([best[g.name] for g in GENES], dtype=float))
        else:
            st.error("Run scripts/evolve_adversary.py first to get a winner to load.")
    if c3.button("Randomize", use_container_width=True):
        rng = np.random.default_rng()
        lo = np.array([g.low for g in GENES])
        hi = np.array([g.high for g in GENES])
        _apply_preset(rng.uniform(lo, hi))


def _apply_preset(genome: np.ndarray) -> None:
    values = decode_genes(genome)
    for g in GENES:
        if g.name == "typology_template":
            continue
        st.session_state[f"gene_{g.name}"] = values[g.name]
    st.session_state["gene_typology_template_name"] = TEMPLATES[int(values["typology_template"])]
    st.rerun()


def _step_for(gene) -> float:
    """A slider step proportional to the gene's range, so wide-range genes
    (dormancy_h: 0-168) aren't stuck at a thousand tiny increments."""
    span = gene.high - gene.low
    if span <= 1:
        return 0.01
    if span <= 10:
        return 0.1
    if span <= 100:
        return 1.0
    return 5.0


def _render_sliders() -> None:
    st.markdown(
        "**Template** — the routing shape the strategy executes "
        "(same vocabulary the detector was trained to recognise)"
    )
    default_template = st.session_state.get(
        "gene_typology_template_name", TEMPLATES[int(_NAIVE["typology_template"])]
    )
    st.selectbox(
        "typology_template", TEMPLATES,
        index=TEMPLATES.index(default_template), key="gene_typology_template_name",
    )

    cols = st.columns(3)
    for col, (group_name, names) in zip(cols, _GROUPS.items()):
        with col:
            st.markdown(f"**{group_name}**")
            for name in names:
                gene = _GENE_BY_NAME[name]
                key = f"gene_{name}"
                default = st.session_state.get(key, _NAIVE[name])
                if gene.is_int:
                    st.slider(name, int(gene.low), int(gene.high), int(default), step=1, key=key, help=gene.note)
                else:
                    st.slider(
                        name, float(gene.low), float(gene.high), float(default),
                        step=_step_for(gene), key=key, help=gene.note,
                    )


def _current_genome() -> np.ndarray:
    values = {
        name: st.session_state.get(f"gene_{name}", _NAIVE[name])
        for name in _GENE_BY_NAME
        if name != "typology_template"
    }
    template_name = st.session_state.get(
        "gene_typology_template_name", TEMPLATES[int(_NAIVE["typology_template"])]
    )
    values["typology_template"] = float(TEMPLATES.index(template_name))
    return np.array([values[g.name] for g in GENES], dtype=np.float64)


def _render_result(outcome, sub, scores: np.ndarray, tau: float) -> None:
    if sub.empty:
        st.error("This genome decoded to zero transactions — try different values.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fitness", f"{outcome.fitness:.3f}")
    c2.metric(
        "Completion", f"{outcome.completion:.1%}",
        help="Fraction of the goal amount delivered by the deadline",
    )
    c3.metric(
        "Evasion", f"{outcome.evasion:.1%}",
        help="Amount-weighted fraction that went undetected",
    )
    c4.metric("Flagged", f"{outcome.n_flagged}/{outcome.n_transactions} transactions")

    flagged = scores >= tau
    fig = go.Figure(
        go.Scatter(
            x=sub["timestamp"], y=sub["amount_paid"], mode="markers",
            marker=dict(size=12, color=["#c53030" if f else "#2f855a" for f in flagged]),
            text=[f"score={s:.3f}" for s in scores],
            hovertemplate="%{x}<br>$%{y:,.0f}<br>%{text}<extra></extra>",
        )
    )
    fig.add_hline(
        y=10000, line_dash="dash", line_color="gray", annotation_text="$10,000 reporting threshold"
    )
    fig.update_layout(
        xaxis_title="time", yaxis_title="amount ($)",
        height=340, margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🔴 flagged by the detector &nbsp;&nbsp; 🟢 passed through unnoticed")

    table = sub[["timestamp", "from_account", "to_account", "amount_paid", "payment_format"]].copy()
    table["detector_score"] = scores.round(4)
    table["flagged"] = flagged
    st.dataframe(table, hide_index=True, use_container_width=True)

    if outcome.completion < 0.95:
        st.warning(
            f"Completion {outcome.completion:.1%} is below the 95% floor — "
            "fitness is forced to 0 regardless of evasion. The adversary's goal "
            "is to launder the money, not most of it."
        )
