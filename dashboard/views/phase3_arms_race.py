"""Phase 3 walkthrough: the closed loop, and who is winning the arms race.

Everything on this page is computed from reports/phase3_metrics.json -- no
number is hard-coded -- so it stays true to whatever the last run produced.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from dashboard import data


def render() -> None:
    st.title("🥊 Phase 3 — The Arms Race")
    st.caption(
        "Objective 4: feed the adversary's evasive strategies back into the detector, "
        "retrain, and measure recall against the adversary across rounds."
    )

    res = data.phase3_metrics()
    if res is None:
        st.warning(
            "No arms-race results yet. Run the closed loop first "
            "(about half an hour at full size):\n\n"
            "```bash\nuv run python scripts/run_arena.py\n```"
        )
        return

    dets, advs = res["detectors"], res["adversaries"]
    cfg = res["settings"]
    n_rounds = len(advs)

    st.markdown(
        f"""
One **round** is a fight and a lesson: the current detector **D_r** is attacked by
an adversary **A_r** evolved against it, the adversary's best strategies are fed
back into the detector's training data, and the detector is retrained into
**D_(r+1)**. This run: **{n_rounds} rounds**, so {n_rounds + 1} detectors and
{n_rounds} adversaries, in {res['runtime_minutes']:.0f} minutes.
"""
    )

    _headline(dets, advs)
    st.divider()
    _arms_race(res, dets, advs)
    st.divider()
    _crossplay(res)
    st.divider()
    _holdout(dets)
    st.divider()
    _feedback_and_genomes(res, dets, advs, cfg)


def _headline(dets: list[dict], advs: list[dict]) -> None:
    own = [a["champion_vs_own"] for a in advs]
    h0, hN = dets[0]["holdout"]["recall_at_budget"], dets[-1]["holdout"]["recall_at_budget"]
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Detector recall vs. its contemporary adversary",
        f"{own[-1]['detector_recall']:.1%}",
        delta=f"{own[-1]['detector_recall'] - own[0]['detector_recall']:+.1%} since round 0",
        help="D_r against the champion of A_r, on held-out episodes from a later period. "
        "Low means the adversary is winning.",
    )
    c2.metric(
        "Adversary success rate",
        f"{own[-1]['success_rate']:.0%}",
        delta=f"{own[-1]['success_rate'] - own[0]['success_rate']:+.0%} since round 0",
        delta_color="inverse",
        help="Fraction of episodes where the money arrived and over half its value went unflagged.",
    )
    c3.metric(
        "Clean-holdout recall @ 0.5% budget",
        f"{hN:.1%}",
        delta=f"{hN - h0:+.1%} since round 0",
        help="What the detector pays for robustness. It never sees adversary data in evaluation.",
    )


def _arms_race(res: dict, dets: list[dict], advs: list[dict]) -> None:
    st.subheader("The arms-race curve")
    ev = res.get("evaluation", {})
    st.caption(
        "Every line is measured on the same fixed episodes, drawn from a later period than "
        "any the detector trained on, so a change between rounds means the detector or "
        "adversary changed — not the dice."
        + (f" **{ev['episodes']} episodes per point; the red bars are 95% bootstrap intervals "
           "over episodes.** A strategy that sends one transaction is a coin flip per episode, "
           "so with 30 episodes the error bar is wider than most round-to-round changes — "
           "which is why this was re-measured." if ev else "")
    )
    own = [a["champion_vs_own"] for a in advs]
    ar = [a["round"] for a in advs]
    dr = [d["round"] for d in dets]
    fig = go.Figure()
    has_ci = all("ci" in s for s in own)
    error_y = None
    if has_ci:
        err_hi = [s["ci"]["detector_recall"][1] - s["detector_recall"] for s in own]
        err_lo = [s["detector_recall"] - s["ci"]["detector_recall"][0] for s in own]
        error_y = dict(type="data", symmetric=False, array=err_hi, arrayminus=err_lo, thickness=1.5)
    fig.add_trace(go.Scatter(x=ar, y=[s["detector_recall"] for s in own], mode="lines+markers",
                             name="recall vs. contemporary adversary (D_r vs A_r)",
                             error_y=error_y, line=dict(color="#c53030", width=3)))
    fig.add_trace(go.Scatter(x=ar, y=[s["success_rate"] for s in own], mode="lines+markers",
                             name="adversary success rate", line=dict(color="#2f855a", width=2)))
    fig.add_trace(go.Scatter(x=dr, y=[d["naive"]["detector_recall"] for d in dets], mode="lines+markers",
                             name="recall vs. the fixed naive strategy",
                             line=dict(color="#718096", width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=dr, y=[d["random"]["detector_recall"] for d in dets], mode="lines+markers",
                             name="recall vs. unseen random strategies",
                             line=dict(color="#2b6cb0", width=2, dash="dash")))
    fig.update_layout(
        xaxis=dict(title="round r", dtick=1), yaxis=dict(title="rate", range=[0, 1.05]),
        height=400, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.25),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "**Red** answers *can the adversary still win?* **Grey and blue** are the generalisation "
        "probes: strategies the detector was never shown. If feedback only taught it to recognise "
        "the exact strategies it saw, these stay flat; if it learned something general, they rise."
    )


def _crossplay(res: dict) -> None:
    st.subheader("Cross-play: every detector against every adversary")
    M = np.array(res["crossplay"]["recall"])
    n_d, n_a = M.shape
    cx = res["crossplay"]
    hover = None
    if "recall_ci_low" in cx:
        lo, hi = np.array(cx["recall_ci_low"]), np.array(cx["recall_ci_high"])
        hover = [[f"D_{i} vs A_{j}: {M[i, j]:.2f}  (95% CI {lo[i, j]:.2f}-{hi[i, j]:.2f})"
                  for j in range(n_a)] for i in range(n_d)]
    fig = go.Figure(go.Heatmap(
        z=M, x=[f"A_{j}" for j in range(n_a)], y=[f"D_{i}" for i in range(n_d)],
        zmin=0, zmax=1, colorscale="RdYlGn", text=[[f"{v:.2f}" for v in row] for row in M],
        texttemplate="%{text}", colorbar=dict(title="detector<br>recall"),
        customdata=hover, hovertemplate="%{customdata}<extra></extra>" if hover else None,
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=60 * n_d + 120, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="adversary generation (champion)", yaxis_title="detector")
    st.plotly_chart(fig, width="stretch")
    st.markdown(
        """
A single curve cannot tell a detector that is genuinely getting harder to fool from one
that is merely chasing whichever strategy it was last shown. The matrix can. Green means
the detector catches the strategy, red means it slips through.

- **The diagonal** (D_r vs A_r) is the live fight — can the adversary still win?
- **The row just below it** (D_(r+1) vs A_r) is the lesson — did feedback work on the
  strategy it was shown?
- **The rest of the lower triangle** (D_i vs A_j, i > j) is *memory* — does an old evasion
  stay caught, or get forgotten to make room for the new one?
- **The upper triangle** (D_i vs A_j, i < j) is *progress* — are later adversaries really
  better than earlier ones against a fixed target, or is the race just going in circles?
"""
    )


def _holdout(dets: list[dict]) -> None:
    st.subheader("What robustness costs: the clean holdout")
    st.caption(
        "The holdout is the last quarter of the ledger and never contains adversary data. A "
        "detector that got robust by flagging everything would collapse here."
    )
    dr = [d["round"] for d in dets]
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dr, y=[d["holdout"]["recall_at_budget"] for d in dets],
                                 mode="lines+markers", name="recall @ 0.5% budget",
                                 line=dict(color="#2b6cb0", width=3)))
        fig.add_trace(go.Scatter(x=dr, y=[d["holdout"]["pr_auc"] for d in dets],
                                 mode="lines+markers", name="PR-AUC",
                                 line=dict(color="#2f855a", width=2, dash="dash")))
        fig.update_layout(xaxis=dict(title="round r (detector D_r)", dtick=1),
                          yaxis=dict(range=[0, 1]), height=340,
                          margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(fig, width="stretch")
    with c2:
        names = sorted({k for d in dets for k in d["holdout"]["typology_recall"]})
        fig = go.Figure()
        for name in names:
            fig.add_trace(go.Scatter(
                x=dr, y=[d["holdout"]["typology_recall"].get(name) for d in dets],
                mode="lines+markers", name=name))
        fig.update_layout(xaxis=dict(title="round r (detector D_r)", dtick=1),
                          yaxis=dict(title="recall by typology", range=[0, 1]), height=340,
                          margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(fig, width="stretch")
    st.caption(
        "Left: aggregate. Right: per typology — if feedback teaches the detector one shape by "
        "making it forget another, that typology's line falls away from the rest."
    )


def _feedback_and_genomes(res: dict, dets: list[dict], advs: list[dict], cfg: dict) -> None:
    st.subheader("What was fed back, and what the adversary changed")
    cap = res["row_caps"]["train_rows"]
    st.caption(
        f"Adversary rows are capped at {cfg['injection_cap']:.0%} of the original positive count "
        f"({cap:,} rows), shared equally across rounds, and the detector is always retrained from "
        "the **original** training set plus the pool — never from the previous round's augmented set."
    )
    rows = []
    for d in dets[1:]:
        inj = d["injection"]
        rows.append({
            "detector": f"D_{d['round']}",
            "operations fed back": inj["n_instances"],
            "adversary rows": inj["n_rows"],
            "share of positives": f"{inj.get('share_of_positives', 0):.1%}",
            "trees": d["n_trees"],
        })
    st.dataframe(rows, hide_index=True, width="stretch")

    gene_rows = []
    for a in advs:
        c = a["champion"]
        gene_rows.append({
            "adversary": f"A_{a['round']}", "template": c["typology_template_name"],
            "tranches": c["n_splits"], "threshold_hug": round(c["threshold_hug"], 2),
            "mean_delay_h": round(c["mean_delay_h"], 1), "business_hours": round(c["business_hours_bias"], 2),
            "hops": c["n_hops"], "mule_pool": c["mule_pool_size"], "dormancy_h": round(c["dormancy_h"], 1),
            "GA best fitness": round(a["ga"]["best_fitness"], 3),
            "warm-start seeds": a["ga"]["warm_start_seeds"],
        })
    st.markdown("**The champion strategy each round** — how the adversary adapts to the detector:")
    st.dataframe(gene_rows, hide_index=True, width="stretch")

    img = data.report_image("phase3_gene_drift.png")
    if img:
        with st.expander("Gene drift across rounds (heatmap)"):
            st.image(img, width="stretch")
