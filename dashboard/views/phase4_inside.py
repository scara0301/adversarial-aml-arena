"""Phase 4: inside the arms race -- the flows, what each detector relies on, and
where its decisions fall.

Everything here reads reports/phase4_explain.json (written by
scripts/build_explainability.py) and reports/phase3_metrics.json. No model is
loaded and nothing is recomputed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import data

FLAGGED, MISSED = "#c53030", "#a0aec0"


def render() -> None:
    st.title("🔬 Phase 4 — Inside the Arms Race")
    st.caption(
        "Objective 5: show each round's transaction flow, the adversary's evolved strategy, "
        "and how the detector's decisions shift."
    )
    ex = data.phase4_explain()
    p3 = data.phase3_metrics()
    if ex is None or p3 is None:
        st.warning(
            "No Phase 4 views yet. They are built from a finished Phase 3 run:\n\n"
            "```bash\nuv run python scripts/run_arena.py\n"
            "uv run python scripts/build_explainability.py\n```"
        )
        return

    with st.expander("Why there is no decision-boundary picture on this page", expanded=False):
        st.markdown(
            """
The detector decides in a 74-dimensional feature space. Any 2-D picture of its
"boundary" has to invent the other 72 coordinates, and the usual trick -- project to
two principal components, map a grid back with `inverse_transform`, score it -- scores
feature vectors **no transaction could produce**. The contours it draws belong to the
reconstruction, not to the detector.

So this page shows two things that are true instead:

- **What each detector relies on** — exact TreeSHAP attributions, per feature, for
  every detector D_0 … D_R. A shifting boundary *is* a shift in which evidence counts.
- **Where its decisions fall on real transactions** — the same transactions fixed in
  one plane, coloured by whether each detector flags them. The points never move;
  only the decisions do.
"""
        )

    _flows(ex, p3)
    st.divider()
    _attribution(ex)
    st.divider()
    _projection(ex)


# ------------------------------------------------------------------ flows


def _flows(ex: dict, p3: dict) -> None:
    st.header("1. One round, one operation")
    st.markdown(
        "Each round's champion strategy, decoded on held-out laundering tasks and drawn as "
        "the transfers it actually makes. Score the same transfers with **D_r** (the "
        "detector it was evolved to beat) and then **D_(r+1)** (the detector retrained on "
        "it): the difference is that round's lesson."
    )
    flows = ex["flows"]
    c1, c2, c3 = st.columns([1, 1, 2])
    # plain string options mapped back by dict: format_func on int options trips Streamlit's AppTest
    rounds = {f"A_{f['round']}": i for i, f in enumerate(flows)}
    rnd = flows[rounds[c1.selectbox("Adversary (round)", list(rounds))]]
    if not rnd["episodes"]:
        st.info("This champion produced no transfers on the sample episodes.")
        return
    ops = {f"#{k + 1}": k for k in range(len(rnd["episodes"]))}
    e = ops[c2.selectbox("Operation", list(ops))]
    judges = {f"{j} ({'evolved against' if k == 0 else 'retrained on it'})": j for k, j in enumerate(rnd["judges"])}
    judge = judges[c3.radio("Scored by", list(judges), horizontal=True, key="p4_judge")]
    r = rnd["round"]
    ep = rnd["episodes"][e]
    champ = p3["adversaries"][r]["champion"]

    t = pd.DataFrame(ep["transfers"])
    t["score"] = ep["scores"][judge]["score"]
    t["flagged"] = ep["scores"][judge]["flagged"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Goal", f"${ep['goal']:,.2f}")
    m2.metric("Transfers", len(t))
    vals = {}
    for j in rnd["judges"]:
        f = np.array(ep["scores"][j]["flagged"], dtype=bool)
        vals[j] = float((t["amount"] * f).sum() / t["amount"].sum())
    m3.metric(f"Value flagged by {judge}", f"{vals[judge]:.0%}")
    if len(rnd["judges"]) == 2:
        a, b = rnd["judges"]
        m4.metric(f"{a} → {b}", f"{vals[b]:.0%}", delta=f"{vals[b] - vals[a]:+.0%}")
    st.caption(
        f"Strategy A_{r}: template **{champ['typology_template_name']}**, {champ['n_hops']} hop(s), "
        f"{champ['n_splits']} tranche(s), threshold_hug {champ['threshold_hug']:.2f}, "
        f"dormancy {champ['dormancy_h']:.0f} h. The adversary succeeds when the money arrives "
        "and less than half of its value is flagged."
    )
    st.plotly_chart(_flow_figure(t, ep), width="stretch")
    with st.expander("The transfers"):
        show = t[["hours_from_start", "from", "to", "amount", "format", "score", "flagged"]]
        st.dataframe(show.sort_values("hours_from_start"), width="stretch", hide_index=True)


def _flow_figure(t: pd.DataFrame, ep: dict) -> go.Figure:
    pos = {a: tuple(p) for a, p in ep["layout"].items()}
    agg = (t.groupby(["from", "to"], sort=False)
             .agg(n=("amount", "size"), total=("amount", "sum"), flagged=("flagged", "sum"))
             .reset_index())
    fig = go.Figure()
    wmax = agg["total"].max()
    for _, row in agg.iterrows():
        (x0, y0), (x1, y1) = pos[row["from"]], pos[row["to"]]
        share = row["flagged"] / row["n"]
        colour = FLAGGED if share >= 0.5 else MISSED
        fig.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.1, standoff=11, startstandoff=11,
            arrowwidth=1.2 + 4.0 * row["total"] / wmax, arrowcolor=colour, opacity=0.85,
        )
        fig.add_trace(go.Scatter(
            x=[(x0 + x1) / 2], y=[(y0 + y1) / 2], mode="markers",
            marker=dict(size=10, color=colour, opacity=0.01), showlegend=False,
            hovertemplate=(f"{row['from']} → {row['to']}<br>{int(row['n'])} transfer(s), "
                           f"${row['total']:,.2f}<br>{int(row['flagged'])} flagged<extra></extra>"),
        ))
    names = list(pos)
    role = ["source" if a == ep["src"] else "destination" if a == ep["dst"] else "mule" for a in names]
    colour = {"source": "#2b6cb0", "destination": "#2f855a", "mule": "#718096"}
    fig.add_trace(go.Scatter(
        x=[pos[a][0] for a in names], y=[pos[a][1] for a in names], mode="markers+text",
        marker=dict(size=[22 if r != "mule" else 14 for r in role], color=[colour[r] for r in role],
                    line=dict(width=1, color="white")),
        text=[r if r != "mule" else "" for r in role], textposition="top center",
        hovertext=[f"{a} ({r})" for a, r in zip(names, role)], hoverinfo="text", showlegend=False,
    ))
    for name, c in (("flagged (most transfers on this edge)", FLAGGED), ("missed", MISSED)):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=c, width=4), name=name))
    fig.update_layout(
        height=430, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False, range=[-0.08, 1.08]), yaxis=dict(visible=False, range=[0, 1.08]),
        legend=dict(orientation="h", y=-0.02), plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ------------------------------------------------------------- attribution


def _attribution(ex: dict) -> None:
    st.header("2. What each detector relies on")
    att = ex["attribution"]
    st.markdown(
        "Mean absolute **TreeSHAP** contribution of each feature, in log-odds, for every "
        "detector. TreeSHAP is exact for tree models: a transaction's contributions add up to "
        "its raw score. Read a row left to right to see a feature gain or lose trust as the "
        "adversary pushed on it."
    )
    sets = {f"{k} ({v['n_rows']:,} rows)": k for k, v in att["sets"].items()}
    name = sets[st.radio("Measured on", list(sets), horizontal=True, key="p4_set")]
    s = att["sets"][name]
    M = np.array(s["mean_abs_shap"])
    c1, c2 = st.columns([3, 2])
    fig = go.Figure(go.Heatmap(
        z=M, x=att["detectors"], y=att["features"], colorscale="Viridis",
        text=np.round(M, 2), texttemplate="%{text}", colorbar=dict(title="mean |SHAP|"),
        hovertemplate="%{y} under %{x}: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed"))
    c1.plotly_chart(fig, width="stretch")

    d = pd.DataFrame({"feature": att["features"], "drift": s["drift"]}).sort_values("drift")
    bar = go.Figure(go.Bar(
        x=d["drift"], y=d["feature"], orientation="h",
        marker_color=["#2f855a" if v > 0 else "#c53030" for v in d["drift"]],
        hovertemplate="%{y}: %{x:+.3f}<extra></extra>",
    ))
    bar.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10),
                      title=dict(text=f"change {att['detectors'][0]} → {att['detectors'][-1]}", font=dict(size=13)))
    c2.plotly_chart(bar, width="stretch")

    if name == "adversary operations":
        drift = dict(zip(att["features"], s["drift"]))
        notes = []
        if drift.get("amount_mod_100", 0) > 0:
            notes.append(
                "**`amount_mod_100` rises.** That is the generator's round-number artefact (synthetic "
                "laundering is never round, 21.6% of benign is), and the retrained detectors lean on it "
                "harder. A known limitation, not a discovery — see Limitations."
            )
        if drift.get("snd_age_h", 0) < 0:
            notes.append(
                "**Account age and `threshold_gap` lose weight.** The adversary learned to use mature "
                "accounts and amounts away from reporting thresholds, so that evidence stopped separating it."
            )
        if drift.get("amount_log", 0) > 0:
            notes.append(
                "**`amount_log` roughly doubles** — the retrained detectors learned that the "
                "single-large-transfer family is suspicious by size alone."
            )
        for n in notes:
            st.markdown(f"- {n}")


# ------------------------------------------------------------- projection


def _projection(ex: dict) -> None:
    st.header("3. Where the decisions fall")
    proj = ex["projection"]
    ev = proj["explained_variance"]
    st.markdown(
        "Real transactions only — clean-holdout benign and laundering, plus the adversary's "
        "decoded operations — placed once in a fixed two-component PCA plane (features "
        "log-scaled, then standardised). The plane is a rough map: it explains "
        f"**{sum(ev):.0%}** of the variance. Pick a detector to see which of the same points it flags."
    )
    pts = proj["points"]
    df = pd.DataFrame({"x": pts["x"], "y": pts["y"], "group": pts["group"], "round": pts["adversary_round"]})
    det = st.select_slider("Detector", proj["detectors"], value=proj["detectors"][0])
    df["flagged"] = np.array(pts["flags"][det], dtype=bool)

    c = st.columns(3)
    for col, (grp, label) in zip(c, (("adversary", "adversary transactions"),
                                     ("laundering", "holdout laundering"),
                                     ("benign", "holdout benign"))):
        share = df.loc[df.group == grp, "flagged"].mean()
        col.metric(f"{label} flagged by {det}", f"{share:.1%}" if grp == "benign" else f"{share:.0%}")

    focus = df[df.group != "benign"]
    bx = np.percentile(df.loc[df.group == "benign", "x"], [1, 99])
    by = np.percentile(df.loc[df.group == "benign", "y"], [1, 99])
    xr = [min(focus.x.min(), bx[0]), max(focus.x.max(), bx[1])]
    yr = [min(focus.y.min(), by[0]), max(focus.y.max(), by[1])]

    fig = go.Figure()
    b = df[df.group == "benign"]
    fig.add_trace(go.Scattergl(x=b.x, y=b.y, mode="markers", name="benign (holdout)",
                               marker=dict(size=4, color="#cbd5e0"), hoverinfo="skip"))
    for grp, colour in (("laundering", "#2b6cb0"), ("adversary", "#c53030")):
        g = df[df.group == grp]
        for flagged, sym in ((False, "circle-open"), (True, "circle")):
            h = g[g.flagged == flagged]
            fig.add_trace(go.Scattergl(
                x=h.x, y=h.y, mode="markers", name=f"{grp}, {'flagged' if flagged else 'missed'}",
                marker=dict(size=6, color=colour, symbol=sym),
                customdata=h["round"], hovertemplate=(
                    "adversary A_%{customdata}<extra></extra>" if grp == "adversary"
                    else "holdout laundering<extra></extra>"),
            ))
    pad = lambda r: [r[0] - 0.05 * (r[1] - r[0]), r[1] + 0.05 * (r[1] - r[0])]  # noqa: E731
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(range=pad(xr), title=f"PC1 ({ev[0]:.0%})"),
                      yaxis=dict(range=pad(yr), title=f"PC2 ({ev[1]:.0%})"),
                      legend=dict(orientation="h", y=1.04))
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Axes are framed on the laundering and adversary points; a few far benign hub "
        "transactions fall outside. Step from D_0 to D_1: most of the red open circles fill in, "
        "while the blue (clean laundering) barely changes — the detector learned the adversary "
        "without forgetting the original typologies. After D_1 the picture hardly moves, which "
        "is the plateau the Phase 3 cross-play matrix shows."
    )
