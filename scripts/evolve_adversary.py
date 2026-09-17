"""Evolve the adversary against a trained detector.

Phase 2. Runs the GA, then measures the headline result: how far the detector's
recall falls between a naive, un-evolved laundering strategy and the evolved one.
That gap is the whole point of the phase -- if it is zero, there is nothing for
the arms race to race against.

    uv run python scripts/evolve_adversary.py --detector artifacts/detector_r0.lgb
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

from arena.adversary import fitness as fit
from arena.adversary.decoder import decode
from arena.adversary.ga import GAConfig, distinct_top, evolve, naive_genome
from arena.adversary.genome import GENES, describe, random_genome, to_dict
from arena.config import Config
from arena.data.loader import load_ledger
from arena.detector import threshold_for_budget
from arena.detector.model import Detector
from arena.features import AccountGraph, featurize


def _measure(genome, episodes, detector, graph, tau, rng, fcfg) -> dict:
    """Score one strategy on a fixed episode set, for like-for-like comparison."""
    _, diag = fit.evaluate(np.asarray(genome, float), episodes, detector, graph, tau, rng, fcfg)
    return diag


def _plot_convergence(history, out: Path) -> None:
    gens = [h.generation for h in history]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True)

    ax1.plot(gens, [h.best_fitness for h in history], lw=2, color="#c53030", label="best")
    ax1.plot(gens, [h.mean_fitness for h in history], lw=1.5, ls="--", color="#718096", label="population mean")
    ax1.set_ylabel("fitness")
    ax1.set_title("Adversary evolution")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.25)

    ax2.plot(gens, [h.best_evasion for h in history], lw=2, color="#2b6cb0", label="evasion (best)")
    ax2.plot(gens, [h.best_completion for h in history], lw=2, color="#2f855a", label="completion (best)")
    ax2.plot(gens, [h.diversity for h in history], lw=1.5, ls=":", color="#805ad5", label="population diversity")
    ax2.set_xlabel("generation")
    ax2.set_ylabel("component value")
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _plot_comparison(rows: list[tuple[str, float]], out: Path) -> None:
    names = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colors = ["#718096", "#2b6cb0", "#c53030"][: len(rows)]

    fig, ax = plt.subplots(figsize=(7, 3.4))
    bars = ax.barh(names, values, color=colors)
    for bar, value in zip(bars, values):
        ax.text(min(value + 0.02, 0.9), bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}", va="center", fontsize=10)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("detector recall on the adversary's transactions")
    ax.set_title("Detector recall: naive vs evolved laundering")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    cfg = Config.load()
    adv = cfg.section("adversary")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--detector", default=str(cfg.path("paths.detector")))
    ap.add_argument("--ledger", default=str(cfg.path("paths.synthetic_ledger")))
    ap.add_argument("--pop", type=int, default=adv.get("population", 100))
    ap.add_argument("--generations", type=int, default=adv.get("generations", 40))
    ap.add_argument("--episodes", type=int, default=adv.get("episodes_per_eval", 5))
    ap.add_argument("--seed", type=int, default=adv.get("seed", 424242))
    ap.add_argument("--budget", type=float, default=cfg.get("threshold.alert_budget", 0.005))
    ap.add_argument("--reports", default=str(cfg.path("paths.reports")))
    args = ap.parse_args(argv)

    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)

    print(f"loading detector {args.detector}")
    detector = Detector.load(args.detector)
    print(f"  {detector.booster.num_trees()} trees, {len(detector.features)} features")

    print(f"loading ledger {args.ledger}")
    ledger = load_ledger("parquet", path=args.ledger)
    t0 = time.time()
    graph = AccountGraph.build(ledger, epoch=detector.epoch)
    print(f"  froze background graph: {graph.n_accounts:,} accounts, {len(ledger):,} rows, "
          f"{time.time() - t0:.1f}s")

    # Invariant 3: the operating threshold comes from the frozen clean holdout,
    # not from anything the adversary can influence.
    holdout = load_ledger("parquet", path=cfg.path("paths.holdout_clean"))
    X_holdout = featurize(holdout, graph, detector.window_hours)
    tau = threshold_for_budget(detector.score_features(X_holdout), args.budget)
    print(f"  threshold at a {args.budget:.2%} alert budget: tau = {tau:.6f}")

    ep_cfg = cfg.section("adversary.episode")
    fit_cfg = cfg.section("adversary.fitness")
    fcfg = fit.FitnessConfig(
        alpha=fit_cfg.get("alpha", 1.0),
        beta=fit_cfg.get("beta", 1.0),
        gamma=fit_cfg.get("gamma", 0.1),
        lambda_time=fit_cfg.get("lambda_time", 0.3),
        completion_floor=fit_cfg.get("completion_floor", 0.95),
        fee_rate=cfg.get("adversary.constraints.fee_rate", 0.002),
    )
    amount_goal = ep_cfg.get("amount_goal", 250_000.0)
    deadline = ep_cfg.get("deadline_hours", 336.0)

    ga_cfg = GAConfig(
        population=args.pop,
        generations=args.generations,
        tournament_size=adv.get("tournament_size", 3),
        crossover_prob=adv.get("crossover_prob", 0.7),
        mutation_prob=adv.get("mutation_prob", 0.2),
        elitism=adv.get("elitism", 5),
        episodes_per_eval=args.episodes,
        seed=args.seed,
    )

    print(f"\nevolving: population {ga_cfg.population}, {ga_cfg.generations} generations, "
          f"{ga_cfg.episodes_per_eval} episodes per evaluation "
          f"(~{ga_cfg.population * (ga_cfg.generations + 1) * ga_cfg.episodes_per_eval:,} "
          f"candidate evaluations)")
    print(f"  goal: move ${amount_goal:,.0f} within {deadline:.0f}h; "
          f"completion floor {fcfg.completion_floor:.2f}")

    t0 = time.time()
    best, history, population = evolve(
        detector, graph, tau, ledger, ga_cfg, fcfg, amount_goal, deadline, detector.epoch
    )
    elapsed = time.time() - t0
    print(f"  done in {elapsed / 60:.1f} min")

    # Head-to-head on one fixed, freshly drawn episode set, so the three
    # strategies are compared on identical tasks rather than on their own
    # lucky draws.
    rng = np.random.default_rng(args.seed + 1)
    sampler = fit.EpisodeSampler(ledger, detector.epoch)
    episodes = sampler.sample(20, rng, amount_goal, deadline)

    naive = naive_genome()
    random_pool = [random_genome(np.random.default_rng(args.seed + 100 + i)) for i in range(10)]

    print("\nre-scoring strategies on 20 fresh episodes")
    naive_diag = _measure(naive, episodes, detector, graph, tau, rng, fcfg)
    evolved_diag = _measure(best, episodes, detector, graph, tau, rng, fcfg)
    random_diags = [_measure(g, episodes, detector, graph, tau, rng, fcfg) for g in random_pool]
    random_recall = float(np.mean([d["detector_recall"] for d in random_diags]))
    random_fitness = float(np.mean([d["fitness"] for d in random_diags]))

    rows = [
        ("naive (hand-written)", naive_diag["detector_recall"]),
        ("random (unevolved mean)", random_recall),
        ("evolved (GA best)", evolved_diag["detector_recall"]),
    ]

    print("\nDetector recall on adversary transactions")
    print("-" * 42)
    for name, value in rows:
        print(f"  {name:<26}{value:.4f}")
    print()
    print(f"  naive fitness            {naive_diag['fitness']:.4f}  "
          f"(completion {naive_diag['completion']:.3f}, evasion {naive_diag['evasion']:.3f})")
    print(f"  random fitness           {random_fitness:.4f}")
    print(f"  evolved fitness          {evolved_diag['fitness']:.4f}  "
          f"(completion {evolved_diag['completion']:.3f}, evasion {evolved_diag['evasion']:.3f})")
    print(f"\nbest evolved strategy\n{describe(best)}")

    # A concrete decoded example, for the writeup and the dashboard.
    example = decode(best, episodes[0], np.random.default_rng(7), fcfg.fee_rate)
    example_path = reports / "phase2_best_strategy_example.csv"
    example.to_csv(example_path, index=False)

    diverse = distinct_top(population, 8)
    payload = {
        "phase": 2,
        "detector": str(args.detector),
        "threshold": float(tau),
        "alert_budget": args.budget,
        "runtime_minutes": round(elapsed / 60, 2),
        "ga": {
            "population": ga_cfg.population,
            "generations": ga_cfg.generations,
            "episodes_per_eval": ga_cfg.episodes_per_eval,
            "evaluations": ga_cfg.population * (ga_cfg.generations + 1) * ga_cfg.episodes_per_eval,
            "seed": ga_cfg.seed,
        },
        "fitness_config": {
            "alpha": fcfg.alpha, "beta": fcfg.beta, "gamma": fcfg.gamma,
            "lambda_time": fcfg.lambda_time, "completion_floor": fcfg.completion_floor,
        },
        "episode": {"amount_goal": amount_goal, "deadline_hours": deadline},
        "headline": {
            "detector_recall_naive": naive_diag["detector_recall"],
            "detector_recall_random": random_recall,
            "detector_recall_evolved": evolved_diag["detector_recall"],
            "recall_drop": naive_diag["detector_recall"] - evolved_diag["detector_recall"],
        },
        "naive": naive_diag,
        "evolved": evolved_diag,
        "best_genome": to_dict(best),
        "gene_spec": [{"name": g.name, "low": g.low, "high": g.high, "note": g.note} for g in GENES],
        "diverse_strategies": [to_dict(g) for g in diverse],
        "history": [
            {
                "generation": h.generation,
                "best_fitness": h.best_fitness,
                "mean_fitness": h.mean_fitness,
                "best_completion": h.best_completion,
                "best_evasion": h.best_evasion,
                "diversity": h.diversity,
            }
            for h in history
        ],
    }
    metrics_path = reports / "phase2_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nwrote {metrics_path}")
    print(f"wrote {example_path}  ({len(example)} transactions)")

    _plot_convergence(history, reports / "phase2_convergence.png")
    _plot_comparison(rows, reports / "phase2_recall_comparison.png")
    print(f"wrote figures -> {reports}/phase2_*.png")

    # Population mean, not best-of-generation: with elitism the best individual
    # can plateau from generation 0 (a lucky initial draw is carried forward
    # untouched), so a flat best curve does not mean the GA failed to learn. The
    # mean rising is the signal that selection is actually working.
    improved = history[-1].mean_fitness > history[0].mean_fitness * 1.2
    gap = naive_diag["detector_recall"] - evolved_diag["detector_recall"]
    ok = improved and gap > 0.15 and evolved_diag["completion"] >= fcfg.completion_floor
    print(f"\nPhase 2 gate (fitness improved, recall gap > 0.15, completion >= floor): "
          f"{'PASS' if ok else 'FAIL'}")
    print(f"  best fitness {history[0].best_fitness:.4f} -> {history[-1].best_fitness:.4f}")
    print(f"  mean fitness {history[0].mean_fitness:.4f} -> {history[-1].mean_fitness:.4f}")
    print(f"  recall gap {gap:.4f}")
    print(f"  evolved completion {evolved_diag['completion']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
