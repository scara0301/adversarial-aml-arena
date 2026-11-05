"""Phase 4: seeing inside the arms race.

The synopsis promises a visualisation of "each round's transaction flow, the
adversary's evolved strategy, and the detector's shifting decision boundary".
Two of those can be drawn directly. The third cannot, and this module is about
what to draw instead.

**There is no picture of a 74-dimensional decision boundary.** Any 2-D slice or
projection of it shows a boundary through points that do not exist. The tempting
trick -- project to two principal components, lay a grid over the plane, map the
grid back with ``inverse_transform`` and score it -- scores synthetic transactions
that sit *off* the data manifold (a 74-feature vector reconstructed from two
coordinates is not a transaction anyone could make), and the contours it draws
are an artefact of the reconstruction, not of the detector. So that is not done
here. Two honest substitutes are:

1. **Attribution drift.** For each detector D_0 .. D_R, the mean absolute TreeSHAP
   contribution of every feature, on fixed sets of transactions. This answers the
   question a "shifting boundary" picture is really asking: *which evidence did the
   detector learn to trust more, or less, as the adversary pushed on it?* LightGBM
   computes exact TreeSHAP natively (``pred_contrib=True``), so no extra dependency.
2. **A fixed projection with each round's decisions.** Real transactions -- clean
   holdout laundering, holdout benign, and the adversary's decoded operations --
   are placed once in a fixed 2-D PCA plane, and coloured by whether each round's
   detector flags them. Nothing is synthesised; the points never move. What moves
   is which of them each detector catches, and that *is* the boundary shifting, seen
   only where there is data.

The transaction flows are drawn directly: each round's champion decoded on fixed
held-out episodes, with every transfer scored by the detector it was evolved
against (D_r) and by the detector retrained on it (D_(r+1)).

Everything here is precomputed into ``reports/phase4_explain.json`` so the
dashboard can show it without the trained models on disk.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from arena.adversary.decoder import Episode, decode
from arena.detector.model import Detector
from arena.features import AccountGraph, featurize


# --------------------------------------------------------------- attribution


def mean_abs_shap(detector: Detector, X: pd.DataFrame) -> pd.Series:
    """Mean |TreeSHAP| per feature, in log-odds units, on the rows of ``X``.

    ``pred_contrib`` returns one column per feature plus a final bias column; the
    contributions of a row sum to its raw (log-odds) score exactly. That identity
    is what makes these attributions trustworthy rather than approximate, and
    ``tests/test_explain.py`` checks it.
    """
    aligned = X.reindex(columns=detector.features, fill_value=0.0).to_numpy(dtype=np.float64)
    contrib = detector.booster.predict(aligned, pred_contrib=True)
    return pd.Series(np.abs(contrib[:, :-1]).mean(axis=0), index=detector.features)


def attribution_drift(
    detectors: list[Detector], sets: dict[str, pd.DataFrame], top: int = 15
) -> dict:
    """Mean |SHAP| per feature, per detector, per evaluation set.

    ``top`` features are chosen by their largest attribution in any detector on any
    set, so a feature that only becomes important later is not cut off by ranking
    on D_0 alone.
    """
    tables = {
        name: pd.DataFrame({f"D_{i}": mean_abs_shap(d, X) for i, d in enumerate(detectors)})
        for name, X in sets.items()
    }
    peak = pd.concat(tables.values(), axis=1).max(axis=1)
    features = list(peak.sort_values(ascending=False).head(top).index)
    out = {"features": features, "detectors": [f"D_{i}" for i in range(len(detectors))], "sets": {}}
    for name, t in tables.items():
        t = t.loc[features]
        out["sets"][name] = {
            "n_rows": int(len(sets[name])),
            "mean_abs_shap": t.to_numpy().tolist(),
            # change from the first detector to the last, the "what did it learn" column
            "drift": (t.iloc[:, -1] - t.iloc[:, 0]).to_numpy().tolist(),
        }
    return out


# ------------------------------------------------------------- projection


def fixed_projection(
    fit_on: pd.DataFrame, groups: dict[str, pd.DataFrame], seed: int = 0
) -> tuple[dict[str, np.ndarray], list[float]]:
    """Place every group of real transactions in one 2-D PCA plane.

    Fitted once, on ``fit_on``, and never refitted, so a point's position is the
    same in every round. Returns the coordinates per group and the variance the two
    axes explain -- which is reported, because a plane that explains a quarter of
    the variance is a rough map and should be read as one.

    Each feature is put through a signed ``log1p`` before standardising. Volumes,
    counts and account ages span several orders of magnitude, and without the log a
    handful of hub accounts define both principal axes: the first version of this
    plane "explained 50% of the variance" and crushed every transaction that mattered
    into one corner, because what it explained was the outliers. Standardising alone
    does not fix that -- it rescales a heavy tail, it does not shorten it.
    """
    cols = list(fit_on.columns)

    def prep(X: pd.DataFrame) -> np.ndarray:
        a = X.reindex(columns=cols, fill_value=0.0).to_numpy(dtype=np.float64)
        return np.sign(a) * np.log1p(np.abs(a))

    scaler = StandardScaler().fit(prep(fit_on))
    pca = PCA(n_components=2, random_state=seed).fit(scaler.transform(prep(fit_on)))
    coords = {name: pca.transform(scaler.transform(prep(X))) for name, X in groups.items()}
    return coords, [float(v) for v in pca.explained_variance_ratio_]


# ------------------------------------------------------------------ flows


def flow_layout(edges: pd.DataFrame, src: str, dst: str) -> dict[str, tuple[float, float]]:
    """Left-to-right positions for an operation's accounts.

    Column = hop distance from the source (breadth-first over the operation's own
    transfers), with the destination always in the last column, so a fan-out reads
    as one account spraying right and a cycle's return edge as the one line that
    runs backwards. Rows are spread evenly within each column.
    """
    out_adj: dict[str, set[str]] = {}
    for f, t in zip(edges["from_account"].astype(str), edges["to_account"].astype(str)):
        out_adj.setdefault(f, set()).add(t)

    depth = {src: 0}
    queue = deque([src])
    while queue:
        node = queue.popleft()
        for nxt in sorted(out_adj.get(node, ())):
            if nxt not in depth:
                depth[nxt] = depth[node] + 1
                queue.append(nxt)
    for acct in set(edges["from_account"].astype(str)) | set(edges["to_account"].astype(str)):
        depth.setdefault(acct, 1)  # unreachable from src: place in the middle
    # the destination sits strictly right of every other account; otherwise a mule
    # at the same hop depth would be drawn in the destination's column
    others = [d for a, d in depth.items() if a != dst]
    depth[dst] = max(others) + 1 if others else 1
    last = depth[dst]

    columns: dict[int, list[str]] = {}
    for acct, d in depth.items():
        columns.setdefault(d, []).append(acct)
    pos: dict[str, tuple[float, float]] = {}
    for d, accts in columns.items():
        accts.sort()
        n = len(accts)
        for k, a in enumerate(accts):
            pos[a] = (float(d) / last, (k + 1) / (n + 1))
    return pos


def decoded_flows(
    champions: list[np.ndarray],
    episodes: list[Episode],
    detectors: list[Detector],
    taus: list[float],
    graph: AccountGraph,
    fee_rate: float,
    seed: int,
) -> list[dict]:
    """Each round's champion decoded on each episode, scored by D_r and D_(r+1).

    The pair is the point: D_r is the detector the strategy was evolved to beat, and
    D_(r+1) is the detector retrained on it, so the two colourings of the same
    transfers show the lesson of one round directly.
    """
    rounds = []
    for r, genome in enumerate(champions):
        judges = [r, r + 1] if r + 1 < len(detectors) else [r]
        eps_out = []
        for e, ep in enumerate(episodes):
            sub = decode(genome, ep, np.random.default_rng(seed + 31 * r + e), fee_rate)
            if sub.empty:
                continue
            X = featurize(sub, graph.overlay(sub), detectors[r].window_hours)
            scored = {}
            for j in judges:
                s = detectors[j].score_features(X)
                scored[f"D_{j}"] = {"score": s.round(4).tolist(), "flagged": (s >= taus[j]).tolist()}
            pos = flow_layout(sub, ep.src_account, ep.dst_account)
            hours = ((pd.to_datetime(sub["timestamp"]) - ep.epoch).dt.total_seconds() / 3600.0 - ep.start_hour)
            eps_out.append({
                "src": ep.src_account,
                "dst": ep.dst_account,
                "goal": float(ep.amount_goal),
                "transfers": {
                    "from": sub["from_account"].astype(str).tolist(),
                    "to": sub["to_account"].astype(str).tolist(),
                    "amount": sub["amount_paid"].round(2).tolist(),
                    "hours_from_start": hours.round(2).tolist(),
                    "format": sub["payment_format"].astype(str).tolist(),
                },
                "scores": scored,
                "layout": {a: list(p) for a, p in pos.items()},
            })
        rounds.append({"round": r, "judges": [f"D_{j}" for j in judges], "episodes": eps_out})
    return rounds
