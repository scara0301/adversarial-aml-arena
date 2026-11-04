"""Run the closed-loop arena: detector and adversary retrained against each other.

Phase 3. Starting from the Phase 1 detector, each round evolves an adversary
against the current detector, feeds its evasive strategies back into the
detector's training set (capped, from the original training rows), and retrains.
Writes the arms-race curve, the cross-play matrix, and per-round metrics.

    uv run python scripts/run_arena.py

A full run (5 rounds, population 100, 40 generations) takes roughly half an hour;
``--rounds 2 --pop 30 --generations 10`` gives a quick look in a few minutes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from arena.adversary.fitness import FitnessConfig
from arena.adversary.ga import GAConfig
from arena.config import Config
from arena.data.loader import load_ledger
from arena.detector.model import Detector
from arena.loop import ArenaSettings, run_arena
from arena.loop.gate import gate
from arena.loop.report import plot_all, print_summary


def main(argv: list[str] | None = None) -> int:
    cfg = Config.load()
    arena = cfg.section("arena")
    adv = cfg.section("adversary")

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector", default=str(cfg.path("paths.detector")), help="the round-0 detector")
    ap.add_argument("--ledger", default=str(cfg.path("paths.synthetic_ledger")))
    ap.add_argument("--rounds", type=int, default=arena.get("rounds", 5))
    ap.add_argument("--pop", type=int, default=adv.get("population", 100))
    ap.add_argument("--generations", type=int, default=adv.get("generations", 40))
    ap.add_argument("--episodes", type=int, default=adv.get("episodes_per_eval", 5))
    ap.add_argument("--cap", type=float, default=arena.get("injection_cap", 0.20))
    ap.add_argument("--top-k", type=int, default=arena.get("top_k", 8))
    ap.add_argument("--no-warm-start", action="store_true", help="evolve each round's GA from scratch")
    ap.add_argument("--fixed-goal", type=float, default=None,
                    help="use one fixed goal amount instead of the configured range (reproduces "
                    "the first run, in which a round $250,000 goal let the GA exploit the "
                    "generator's round-number artefact)")
    ap.add_argument("--seed", type=int, default=adv.get("seed", 424242))
    ap.add_argument("--budget", type=float, default=cfg.get("threshold.alert_budget", 0.005))
    ap.add_argument("--out-dir", default=str(cfg.path("arena.artifacts")))
    ap.add_argument("--reports", default=str(cfg.path("paths.reports")))
    args = ap.parse_args(argv)

    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)

    print(f"round-0 detector: {args.detector}")
    detector0 = Detector.load(args.detector)
    print(f"loading ledger {args.ledger}")
    ledger = load_ledger("parquet", path=args.ledger)
    holdout = load_ledger("parquet", path=cfg.path("paths.holdout_clean"))

    fit_cfg = cfg.section("adversary.fitness")
    fitness_cfg = FitnessConfig(
        alpha=fit_cfg.get("alpha", 1.0), beta=fit_cfg.get("beta", 1.0),
        gamma=fit_cfg.get("gamma", 0.1), lambda_time=fit_cfg.get("lambda_time", 0.3),
        completion_floor=fit_cfg.get("completion_floor", 0.95),
        fee_rate=cfg.get("adversary.constraints.fee_rate", 0.002),
    )
    ga_cfg = GAConfig(
        population=args.pop, generations=args.generations,
        tournament_size=adv.get("tournament_size", 3),
        crossover_prob=adv.get("crossover_prob", 0.7),
        mutation_prob=adv.get("mutation_prob", 0.2),
        elitism=adv.get("elitism", 5), episodes_per_eval=args.episodes, seed=args.seed,
    )
    settings = ArenaSettings(
        rounds=args.rounds, injection_cap=args.cap, top_k=args.top_k,
        instances_per_strategy=arena.get("instances_per_strategy", 6),
        warm_start=not args.no_warm_start and arena.get("warm_start", True),
        eval_episodes=arena.get("eval_episodes", 30),
        n_random_probes=arena.get("n_random_probes", 10),
        budget=args.budget, seed=cfg.get("synthetic.seed", 20260917),
    )
    ep_cfg = cfg.section("adversary.episode")
    goal_range = arena.get("amount_goal_range")
    amount_goal = (
        args.fixed_goal if args.fixed_goal is not None
        else tuple(float(x) for x in goal_range) if goal_range
        else ep_cfg.get("amount_goal", 250_000.0)
    )
    progress_path = reports / "phase3_progress.json"

    print(f"\nclosed loop: {settings.rounds} rounds, population {ga_cfg.population}, "
          f"{ga_cfg.generations} generations, feedback cap {settings.injection_cap:.0%}, "
          f"warm start {'on' if settings.warm_start else 'off'}")

    result = run_arena(
        ledger, holdout, detector0, settings, ga_cfg, fitness_cfg,
        amount_goal=amount_goal,
        deadline_hours=ep_cfg.get("deadline_hours", 336.0),
        train_frac=cfg.get("detector.train_frac", 0.60),
        valid_frac=cfg.get("detector.valid_frac", 0.15),
        lgb_params=cfg.section("detector.lightgbm"),
        out_dir=args.out_dir,
        on_round=lambda r: progress_path.write_text(json.dumps(r.to_json(), indent=2, default=float), encoding="utf-8"),
    )

    res = result.to_json()
    metrics_path = reports / "phase3_metrics.json"
    metrics_path.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    progress_path.unlink(missing_ok=True)
    figures = plot_all(res, reports)

    print()
    print_summary(res)

    print(f"\nwrote {metrics_path}")
    print(f"wrote {', '.join(p.name for p in figures)}")
    print(f"detectors saved to {args.out_dir}   ({result.cpu_minutes:.1f} CPU-min, "
          f"{result.runtime_minutes:.1f} min wall-clock -- wall time includes any time the machine slept)")

    print("\nPhase 3 gate")
    checks = gate(res)
    for label, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}\n         {detail}")
    ok_all = all(ok for _, ok, _ in checks)
    print(f"\nPhase 3 gate: {'PASS' if ok_all else 'FAIL'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
