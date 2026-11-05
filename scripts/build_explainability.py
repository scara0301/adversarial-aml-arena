"""Precompute the Phase 4 views: attribution drift, the fixed projection, the flows.

Reads the detectors saved by ``run_arena.py`` (``artifacts/arena/``) and the
champions in ``reports/phase3_metrics.json``. Writes ``reports/phase4_explain.json``
plus two figures. Retrains nothing and evolves nothing.

    uv run python scripts/build_explainability.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from arena.adversary.fitness import EpisodeSampler
from arena.config import Config
from arena.data.loader import load_ledger
from arena.detector.model import Detector
from arena.explain import attribution_drift, decoded_flows, fixed_projection
from arena.adversary.decoder import decode
from arena.features import AccountGraph, featurize
from arena.loop.reevaluate import genome_from_dict


def plot_attribution(drift: dict, out: Path) -> None:
    sets = list(drift["sets"])
    fig, axes = plt.subplots(1, len(sets), figsize=(5.2 * len(sets), 0.42 * len(drift["features"]) + 1.8),
                             sharey=True)
    for ax, name in zip(np.atleast_1d(axes), sets):
        M = np.array(drift["sets"][name]["mean_abs_shap"])
        im = ax.imshow(M, cmap="viridis", aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if M[i, j] < 0.6 * M.max() else "black")
        ax.set_xticks(range(M.shape[1]), drift["detectors"])
        ax.set_title(f"{name}  (n={drift['sets'][name]['n_rows']:,})", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.04, label="mean |SHAP| (log-odds)")
    np.atleast_1d(axes)[0].set_yticks(range(len(drift["features"])), drift["features"])
    fig.suptitle("What each detector relies on: mean |TreeSHAP| per feature, D_0 -> D_R", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_projection(proj: dict, out: Path) -> None:
    dets = proj["detectors"]
    pts = proj["points"]
    x, y = np.array(pts["x"]), np.array(pts["y"])
    g = np.array(pts["group"])
    cols = 3
    rows = int(np.ceil(len(dets) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.3 * rows), sharex=True, sharey=True)
    for k, (ax, name) in enumerate(zip(axes.ravel(), dets)):
        f = np.array(pts["flags"][name], dtype=bool)
        ax.scatter(x[g == "benign"], y[g == "benign"], s=3, c="#cbd5e0", label="benign (holdout)")
        for grp, c in (("laundering", "#2b6cb0"), ("adversary", "#c53030")):
            m = g == grp
            ax.scatter(x[m & ~f], y[m & ~f], s=10, facecolors="none", edgecolors=c, linewidths=0.6,
                       label=f"{grp}, missed")
            ax.scatter(x[m & f], y[m & f], s=10, c=c, label=f"{grp}, flagged")
        ax.set_title(f"{name}: flags {f[g == 'adversary'].mean():.0%} of adversary, "
                     f"{f[g == 'laundering'].mean():.0%} of holdout laundering", fontsize=9)
        if k == 0:
            ax.legend(fontsize=7, loc="best", markerscale=2)
    for ax in axes.ravel()[len(dets):]:
        ax.axis("off")
    # frame the transactions that matter: a few far benign points should not set the scale
    focus = g != "benign"
    xs = np.concatenate([x[focus], np.percentile(x[~focus], [1, 99])])
    ys = np.concatenate([y[focus], np.percentile(y[~focus], [1, 99])])
    pad_x, pad_y = 0.05 * np.ptp(xs), 0.05 * np.ptp(ys)
    axes.ravel()[0].set_xlim(xs.min() - pad_x, xs.max() + pad_x)
    axes.ravel()[0].set_ylim(ys.min() - pad_y, ys.max() + pad_y)
    ev = proj["explained_variance"]
    fig.suptitle("The same real transactions, fixed in one plane; only each detector's decisions change  "
                 f"(the plane explains {sum(ev):.0%} of the variance)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    cfg = Config.load()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", default=str(cfg.path("paths.synthetic_ledger")))
    ap.add_argument("--det-dir", default=str(cfg.path("arena.artifacts")))
    ap.add_argument("--reports", default=str(cfg.path("paths.reports")))
    ap.add_argument("--benign-sample", type=int, default=3000)
    ap.add_argument("--adv-episodes", type=int, default=40, help="episodes per champion for SHAP/projection")
    ap.add_argument("--flow-episodes", type=int, default=4)
    args = ap.parse_args(argv)
    t0 = time.time()

    reports = Path(args.reports)
    res = json.loads((reports / "phase3_metrics.json").read_text(encoding="utf-8"))
    s = res["settings"]
    n_det = len(res["detectors"])
    detectors = [Detector.load(Path(args.det_dir) / f"detector_r{i}.lgb") for i in range(n_det)]
    taus = [d["tau"] for d in res["detectors"]]
    champions = [genome_from_dict(a["champion"]) for a in res["adversaries"]]
    fee_rate = cfg.get("adversary.constraints.fee_rate", 0.0)
    wh = detectors[0].window_hours

    ledger = load_ledger("parquet", path=args.ledger)
    holdout = load_ledger("parquet", path=cfg.path("paths.holdout_clean"))
    graph = AccountGraph.build(ledger, epoch=detectors[0].epoch)
    print(f"loaded {n_det} detectors, {len(champions)} champions, graph of {graph.n_accounts:,} accounts")

    # ---- three fixed sets of real transactions
    pos = holdout[holdout["is_laundering"] == 1]
    neg = holdout[holdout["is_laundering"] == 0].sample(n=args.benign_sample, random_state=s["seed"])
    X_pos, X_neg = featurize(pos, graph, wh), featurize(neg, graph, wh)

    goal = tuple(res["episode"]["amount_goal"])
    deadline = res["episode"]["deadline_hours"]
    w0, w1 = res["windows_hours"]["eval"]
    sampler = EpisodeSampler(ledger, detectors[0].epoch)
    eps = sampler.sample(args.adv_episodes, np.random.default_rng(s["seed"] + 20), goal, deadline,
                         start_range=(w0, w1 - deadline))
    adv_X, adv_round = [], []
    for j, genome in enumerate(champions):
        for e, ep in enumerate(eps):
            sub = decode(genome, ep, np.random.default_rng(s["seed"] + 1000 * j + e), fee_rate)
            if sub.empty:
                continue
            adv_X.append(featurize(sub, graph.overlay(sub), wh))
            adv_round += [j] * len(sub)
    X_adv = pd.concat(adv_X, ignore_index=True)
    print(f"sets: {len(X_pos):,} holdout laundering, {len(X_neg):,} holdout benign, "
          f"{len(X_adv):,} adversary transactions")

    # ---- 1. attribution drift
    drift = attribution_drift(detectors, {
        "adversary operations": X_adv, "holdout laundering": X_pos, "holdout benign": X_neg,
    })

    # ---- 2. the fixed projection, with each detector's decisions
    coords, ev = fixed_projection(pd.concat([X_neg, X_pos], ignore_index=True),
                                  {"benign": X_neg, "laundering": X_pos, "adversary": X_adv})
    groups = (["benign"] * len(X_neg)) + (["laundering"] * len(X_pos)) + (["adversary"] * len(X_adv))
    X_all = pd.concat([X_neg, X_pos, X_adv], ignore_index=True)
    xy = np.vstack([coords["benign"], coords["laundering"], coords["adversary"]])
    flags = {f"D_{i}": (d.score_features(X_all) >= taus[i]).astype(int).tolist() for i, d in enumerate(detectors)}
    projection = {
        "detectors": [f"D_{i}" for i in range(n_det)],
        "explained_variance": ev,
        "points": {
            "x": np.round(xy[:, 0], 3).tolist(), "y": np.round(xy[:, 1], 3).tolist(),
            "group": groups, "adversary_round": ([-1] * (len(X_neg) + len(X_pos))) + adv_round,
            "flags": flags,
        },
    }

    # ---- 3. flows
    flow_eps = sampler.sample(args.flow_episodes, np.random.default_rng(s["seed"] + 40), goal, deadline,
                              start_range=(w0, w1 - deadline))
    flows = decoded_flows(champions, flow_eps, detectors, taus, graph, fee_rate, s["seed"])

    out = {
        "phase": 4,
        "source": {"metrics": "reports/phase3_metrics.json", "detectors": args.det_dir},
        "attribution": drift,
        "projection": projection,
        "flows": flows,
    }
    out_path = reports / "phase4_explain.json"
    out_path.write_text(json.dumps(out, default=float), encoding="utf-8")
    plot_attribution(drift, reports / "phase4_attribution_drift.png")
    plot_projection(projection, reports / "phase4_projection.png")

    # ---- summary
    print(f"\nprojection plane explains {sum(ev):.1%} of the variance ({ev[0]:.1%} + {ev[1]:.1%})")
    g = np.array(groups)
    print(f"{'detector':>8} | {'adversary flagged':>17} {'laundering flagged':>18} {'benign flagged':>14}")
    for name, f in flags.items():
        f = np.array(f, dtype=bool)
        print(f"{name:>8} | {f[g == 'adversary'].mean():>17.1%} {f[g == 'laundering'].mean():>18.1%} "
              f"{f[g == 'benign'].mean():>14.2%}")
    adv = drift["sets"]["adversary operations"]
    order = np.argsort(adv["drift"])
    print("\nbiggest attribution changes D_0 -> D_R on adversary operations (mean |SHAP|, log-odds):")
    for k in list(order[::-1][:5]) + list(order[:3]):
        print(f"   {drift['features'][k]:<24} {adv['mean_abs_shap'][k][0]:.3f} -> {adv['mean_abs_shap'][k][-1]:.3f}"
              f"   ({adv['drift'][k]:+.3f})")
    print(f"\nwrote {out_path} ({out_path.stat().st_size / 1e3:.0f} kB), phase4_attribution_drift.png, "
          f"phase4_projection.png  in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
