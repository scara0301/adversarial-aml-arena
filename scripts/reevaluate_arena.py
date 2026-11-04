"""Re-measure a finished arms race with more episodes and bootstrap error bars.

Phase 3 follow-up. Reads ``reports/phase3_metrics.json`` and the detectors saved
by ``run_arena.py``, re-scores every (detector, champion) pair on a larger fixed
set of held-out episodes, and rewrites the metrics with 95% intervals. No GA is
re-run: same detectors, same champions, only the measurement is sharper.

    uv run python scripts/reevaluate_arena.py --episodes 200

Why this exists: see arena/loop/reevaluate.py. In short, a strategy that sends a
couple of transactions is a coin flip per episode, so 30 episodes cannot resolve
round-to-round differences, and the GA's own "best fitness" is a winner's curse.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from arena.adversary.fitness import FitnessConfig
from arena.config import Config
from arena.data.loader import load_ledger
from arena.detector.model import Detector
from arena.loop.gate import gate
from arena.loop.reevaluate import reevaluate
from arena.loop.report import plot_all, print_summary


def main(argv: list[str] | None = None) -> int:
    cfg = Config.load()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap resamples")
    ap.add_argument("--ledger", default=str(cfg.path("paths.synthetic_ledger")))
    ap.add_argument("--det-dir", default=str(cfg.path("arena.artifacts")))
    ap.add_argument("--reports", default=str(cfg.path("paths.reports")))
    args = ap.parse_args(argv)

    metrics_path = Path(args.reports) / "phase3_metrics.json"
    res = json.loads(metrics_path.read_text(encoding="utf-8"))
    n_det = len(res["detectors"])
    detectors = [Detector.load(Path(args.det_dir) / f"detector_r{i}.lgb") for i in range(n_det)]
    print(f"re-measuring {n_det} detectors x {len(res['adversaries'])} champions on "
          f"{args.episodes} episodes (was {res['settings']['eval_episodes']})")

    ledger = load_ledger("parquet", path=args.ledger)
    fit_cfg = cfg.section("adversary.fitness")
    fitness_cfg = FitnessConfig(
        alpha=fit_cfg.get("alpha", 1.0), beta=fit_cfg.get("beta", 1.0),
        gamma=fit_cfg.get("gamma", 0.1), lambda_time=fit_cfg.get("lambda_time", 0.3),
        completion_floor=fit_cfg.get("completion_floor", 0.95),
        fee_rate=cfg.get("adversary.constraints.fee_rate", 0.0),
    )

    new = reevaluate(res, ledger, detectors, fitness_cfg, n_episodes=args.episodes, n_boot=args.boot)
    metrics_path.write_text(json.dumps(new, indent=2, default=float), encoding="utf-8")
    plot_all(new, Path(args.reports))

    print()
    print_summary(new)
    print(f"\nrewrote {metrics_path} and the four figures")

    print("\nPhase 3 gate")
    checks = gate(new)
    for label, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}\n         {detail}")
    ok_all = all(ok for _, ok, _ in checks)
    print(f"\nPhase 3 gate: {'PASS' if ok_all else 'FAIL'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
