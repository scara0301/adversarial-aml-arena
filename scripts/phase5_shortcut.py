"""Phase 5, step 4: how much of the IBM detector's weakness is one dataset shortcut?

88% of IBM HI-Small laundering is paid by ACH, against 12% of benign traffic, and
the IBM-trained detector leans on it: even the hand-written naive strategy, paid
by cheque or card, goes unflagged. This script measures that directly. Every
strategy is scored, on the same held-out episodes with the same decode seeds, by

* **D_0** -- the IBM detector from ``train_detector.py``, and
* **D_0 without format** -- the same model retrained without the two payment
  format features, thresholded by the same alert-budget policy.

Strategies: the naive genome; the naive genome paid by ACH (what IBM laundering
looks like); the Phase 2 champion; and, if the IBM closed loop has run, each
round's champion A_0 .. A_R.

    ARENA_CONFIG=config/ibm.yaml uv run python scripts/phase5_shortcut.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from arena.adversary import fitness as fit
from arena.adversary.ga import naive_genome
from arena.adversary.genome import GENE_INDEX
from arena.config import Config
from arena.data.loader import load_ledger
from arena.detector import evaluate, threshold_for_budget, time_split
from arena.detector.model import Detector, downsample_negatives, fit_detector
from arena.features import AccountGraph, featurize
from arena.loop.reevaluate import genome_from_dict

FORMAT_FEATURES = ["format_code", "format_scrutiny"]
SUCCESS_EVASION = 0.5


def score(genome, episodes, detector, graph, tau, cfg, seed) -> dict:
    flagged = txns = wins = 0
    for e, ep in enumerate(episodes):
        o = fit.evaluate_episode(genome, ep, detector, graph, tau, np.random.default_rng(seed + e), cfg)
        flagged += o.n_flagged
        txns += o.n_transactions
        wins += int(o.completion >= cfg.completion_floor and o.evasion > SUCCESS_EVASION)
    return {"detector_recall": flagged / max(txns, 1), "success_rate": wins / len(episodes),
            "transactions_per_episode": txns / len(episodes)}


def main(argv: list[str] | None = None) -> int:
    cfg = Config.load()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--reports", default=str(cfg.path("paths.reports")))
    args = ap.parse_args(argv)
    reports = Path(args.reports)
    if "ibm" not in str(reports):
        print("run with ARENA_CONFIG=config/ibm.yaml")
        return 2
    t0 = time.time()
    budget = cfg.get("threshold.alert_budget", 0.005)

    d0 = Detector.load(cfg.path("paths.detector"))
    ledger = load_ledger("parquet", path=cfg.path("paths.synthetic_ledger"))
    holdout = load_ledger("parquet", path=cfg.path("paths.holdout_clean"))
    graph = AccountGraph.build(ledger, epoch=d0.epoch)
    wh = d0.window_hours
    X_h = featurize(holdout, graph, wh)
    y_h = holdout["is_laundering"].to_numpy(dtype=int)
    print(f"graph and holdout features in {time.time() - t0:.0f}s")

    # the format-blind detector, trained exactly as D_0 was, minus two columns
    split = time_split(ledger, cfg.get("detector.train_frac", 0.60), cfg.get("detector.valid_frac", 0.15))
    ratio = cfg.get("detector.negative_ratio", 25)
    tr, va = downsample_negatives(split.train, ratio, 0), downsample_negatives(split.valid, ratio, 0)
    X_tr, X_va = featurize(tr, graph, wh), featurize(va, graph, wh)
    cols = [c for c in X_tr.columns if c not in FORMAT_FEATURES]
    blind = fit_detector(X_tr[cols], tr["is_laundering"].to_numpy(dtype=int), X_va[cols],
                         va["is_laundering"].to_numpy(dtype=int), window_hours=wh, epoch=d0.epoch,
                         params=cfg.section("detector.lightgbm"))
    judges = {}
    for name, det in (("D_0", d0), ("D_0 without format", blind)):
        s = det.score_features(X_h)
        tau = threshold_for_budget(s, budget)
        m = evaluate(y_h, s, tau, holdout["typology"], budget=budget)
        judges[name] = {"detector": det, "tau": tau,
                        "holdout": {k: m[k] for k in ("recall_at_budget", "pr_auc", "typology_recall")}}
        print(f"{name:<20} holdout recall {m['recall_at_budget']:.3f}  PR-AUC {m['pr_auc']:.3f}")

    # strategies
    naive = naive_genome()
    naive_ach = naive.copy()
    naive_ach[GENE_INDEX["format_mix"]] = 0.0
    strategies = {"naive": naive, "naive, paid by ACH": naive_ach}
    p2 = json.loads((reports / "phase2_metrics.json").read_text(encoding="utf-8"))
    strategies["Phase 2 champion"] = genome_from_dict(p2["best_genome"])
    p3_path = reports / "phase3_metrics.json"
    if p3_path.exists():
        for a in json.loads(p3_path.read_text(encoding="utf-8"))["adversaries"]:
            strategies[f"A_{a['round']} (arms race)"] = genome_from_dict(a["champion"])

    # held-out episodes from the holdout period, fixed for every cell
    ep_cfg = cfg.section("adversary.episode")
    deadline = ep_cfg.get("deadline_hours", 24.0)
    sampler = fit.EpisodeSampler(ledger, d0.epoch)
    h0 = float((holdout["timestamp"].min() - d0.epoch).total_seconds() / 3600)
    h1 = float((holdout["timestamp"].max() - d0.epoch).total_seconds() / 3600)
    goal_range = tuple(float(x) for x in cfg.get("arena.amount_goal_range", [50000, 400000]))
    episodes = sampler.sample(args.episodes, np.random.default_rng(777), goal_range, deadline,
                              start_range=(h0, h1 - deadline))
    fcfg = fit.FitnessConfig(fee_rate=cfg.get("adversary.constraints.fee_rate", 0.0))

    table = {}
    for s_name, g in strategies.items():
        table[s_name] = {j: score(g, episodes, v["detector"], graph, v["tau"], fcfg, seed=9000)
                         for j, v in judges.items()}
        row = "   ".join(f"{j}: recall {r['detector_recall']:.3f}, adversary wins {r['success_rate']:.0%}"
                         for j, r in table[s_name].items())
        print(f"   {s_name:<22} {row}")

    out = {"phase": 5, "episodes": args.episodes, "goal_range": list(goal_range), "deadline_hours": deadline,
           "judges": {j: {"tau": v["tau"], "holdout": v["holdout"]} for j, v in judges.items()},
           "strategies": table}
    path = reports / "phase5_shortcut.json"
    path.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"wrote {path} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
