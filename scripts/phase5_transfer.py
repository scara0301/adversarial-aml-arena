"""Phase 5, step 3: does anything transfer between the synthetic world and IBM?

Two questions, both answered on frozen clean holdouts with the same 0.5%
alert-budget policy as every other number in the project:

1. **Cross-dataset transfer.** The synthetic-trained detector scored on IBM, and
   the IBM-trained detector scored on synthetic data. The threshold is the same
   *policy* applied to the target's own clean holdout (the alert budget is a
   property of the analysts, not of the model), so the comparison is of rankings.
2. **Feature-group ablation.** The detector retrained on each dataset with only
   the per-transaction features (what a PaySim-style dataset, which has no
   account graph, could support), only the relational features, and -- on IBM --
   everything except payment format, which carries 36% of the IBM model's gain
   because 85% of IBM laundering is ACH against 12% of benign traffic.

Run after ``prepare_ibm.py`` and the IBM ``train_detector.py``:

    uv run python scripts/phase5_transfer.py
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

from arena.config import ROOT, Config
from arena.data.loader import load_ledger
from arena.detector import evaluate, threshold_for_budget, time_split
from arena.detector.model import Detector, downsample_negatives, fit_detector
from arena.features import AccountGraph, featurize
from arena.features.pipeline import _transaction_features

DATASETS = {
    "synthetic": {
        "ledger": "data/synthetic/ledger.parquet",
        "holdout": "data/processed/holdout_clean.parquet",
        "detector": "artifacts/detector_r0.lgb",
    },
    "ibm": {
        "ledger": "data/processed/ibm_ledger.parquet",
        "holdout": "data/processed/ibm_holdout_clean.parquet",
        "detector": "artifacts/ibm/detector_r0.lgb",
    },
}
FORMAT_FEATURES = ["format_code", "format_scrutiny"]


def _metrics(y: np.ndarray, s: np.ndarray, typ: pd.Series, budget: float) -> dict:
    tau = threshold_for_budget(s, budget)
    m = evaluate(y, s, tau, typ, budget=budget)
    return {k: m[k] for k in ("recall_at_budget", "pr_auc", "precision_at_budget", "typology_recall")}


def main(argv: list[str] | None = None) -> int:
    cfg = Config.load()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reports", default=str(ROOT / "reports" / "ibm"))
    ap.add_argument("--budget", type=float, default=cfg.get("threshold.alert_budget", 0.005))
    args = ap.parse_args(argv)
    t0 = time.time()
    params = cfg.section("detector.lightgbm")
    ratio = cfg.get("detector.negative_ratio", 25)

    data = {}
    for name, p in DATASETS.items():
        ledger = load_ledger("parquet", path=ROOT / p["ledger"])
        holdout = load_ledger("parquet", path=ROOT / p["holdout"])
        det = Detector.load(ROOT / p["detector"])
        graph = AccountGraph.build(ledger, epoch=det.epoch)
        X_h = featurize(holdout, graph, det.window_hours)
        data[name] = {"ledger": ledger, "holdout": holdout, "detector": det, "graph": graph, "X_h": X_h,
                      "y_h": holdout["is_laundering"].to_numpy(dtype=int), "typ": holdout["typology"]}
        print(f"{name}: {len(ledger):,} rows, holdout {len(holdout):,} ({int(data[name]['y_h'].sum()):,} pos)"
              f"  [{time.time() - t0:.0f}s]")

    # ---- 1. cross-dataset transfer
    transfer = {}
    for d_name, d in data.items():
        for t_name, t in data.items():
            s = d["detector"].score_features(t["X_h"])
            transfer[f"{d_name}->{t_name}"] = _metrics(t["y_h"], s, t["typ"], args.budget)
    print("\ncross-dataset transfer (recall@budget / PR-AUC), rows = trained on, cols = scored on")
    for d_name in data:
        cells = [f"{transfer[f'{d_name}->{t}']['recall_at_budget']:.3f} / {transfer[f'{d_name}->{t}']['pr_auc']:.3f}"
                 for t in data]
        print(f"   {d_name:<10} " + "   ".join(f"{c:>16}" for c in cells))

    # ---- 2. feature-group ablation, retrained on each dataset
    tx_cols = list(_transaction_features(data["ibm"]["holdout"].head(3)).keys())
    ablation = {}
    for name, d in data.items():
        split = time_split(d["ledger"], cfg.get("detector.train_frac", 0.60), cfg.get("detector.valid_frac", 0.15))
        tr = downsample_negatives(split.train, ratio, 0)
        va = downsample_negatives(split.valid, ratio, 0)
        wh = d["detector"].window_hours
        X_tr, X_va = featurize(tr, d["graph"], wh), featurize(va, d["graph"], wh)
        y_tr, y_va = tr["is_laundering"].to_numpy(dtype=int), va["is_laundering"].to_numpy(dtype=int)
        all_cols = list(X_tr.columns)
        groups = {
            "all features": all_cols,
            "per-transaction only": [c for c in all_cols if c in tx_cols],
            "relational only": [c for c in all_cols if c not in tx_cols],
            "all but payment format": [c for c in all_cols if c not in FORMAT_FEATURES],
        }
        ablation[name] = {}
        for g_name, cols in groups.items():
            det = fit_detector(X_tr[cols], y_tr, X_va[cols], y_va, window_hours=wh,
                               epoch=d["graph"].epoch, params=params)
            s = det.score_features(d["X_h"][cols])
            ablation[name][g_name] = {**_metrics(d["y_h"], s, d["typ"], args.budget), "n_features": len(cols)}
        print(f"\nablation on {name} (recall@budget / PR-AUC)")
        for g_name, m in ablation[name].items():
            print(f"   {g_name:<24} ({m['n_features']:>2} features)  {m['recall_at_budget']:.3f} / {m['pr_auc']:.3f}")

    out = {"phase": 5, "budget": args.budget, "transfer": transfer, "ablation": ablation,
           "per_transaction_features": tx_cols}
    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / "phase5_transfer.json"
    path.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    _plot(out, reports / "phase5_transfer_ablation.png")
    print(f"\nwrote {path} and phase5_transfer_ablation.png in {time.time() - t0:.0f}s")
    return 0


def _plot(out: dict, path: Path) -> None:
    names = list(out["ablation"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1, 1.6]})
    M = np.array([[out["transfer"][f"{d}->{t}"]["recall_at_budget"] for t in names] for d in names])
    im = ax1.imshow(M, cmap="RdYlGn", vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            ax1.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=12, fontweight="bold")
    ax1.set_xticks(range(len(names)), [f"scored on {n}" for n in names])
    ax1.set_yticks(range(len(names)), [f"trained on {n}" for n in names])
    ax1.set_title("Cross-dataset transfer: recall @ 0.5% budget", fontsize=10)
    fig.colorbar(im, ax=ax1, fraction=0.046)

    groups = list(out["ablation"][names[0]])
    w = 0.38
    x = np.arange(len(groups))
    for k, (n, c) in enumerate(zip(names, ("#718096", "#2b6cb0"))):
        vals = [out["ablation"][n][g]["recall_at_budget"] for g in groups]
        bars = ax2.bar(x + (k - 0.5) * w, vals, w, label=n, color=c)
        ax2.bar_label(bars, fmt="%.2f", fontsize=8)
    ax2.set_xticks(x, groups, fontsize=9)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("recall @ 0.5% budget")
    ax2.set_title("Retrained on one feature group at a time (frozen clean holdout)", fontsize=10)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
