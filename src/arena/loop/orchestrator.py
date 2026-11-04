"""The closed-loop round orchestrator.

One *round* is a fight and a lesson:

    detector D_r  --evolve-->  adversary A_r  --feedback-->  detector D_{r+1}

``D_0`` is the Phase 1 detector. Each round evolves a fresh adversary against the
current detector, folds the evasive strategies it found back into the detector's
training data, and retrains. After ``R`` rounds there are ``R + 1`` detectors and
``R`` adversaries, and every detector is scored against every adversary -- the
*cross-play matrix* at the heart of the result.

Why a matrix and not just a curve. A single "recall per round" line cannot tell a
detector that is genuinely getting harder to fool from one that is merely chasing
whichever strategy it was last shown. The matrix can:

* the **diagonal** (``D_r`` vs ``A_r``) is the contemporary fight -- whether the
  adversary can still win;
* the **row just below** it (``D_{r+1}`` vs ``A_r``) is whether the feedback
  worked on the strategies it was shown;
* the **rest of the lower triangle** (``D_i`` vs ``A_j``, ``i > j``) is memory --
  whether an old evasion stays caught, or is forgotten to make room for the new;
* the **upper triangle** (``D_i`` vs ``A_j``, ``i < j``) is progress -- whether
  later adversaries really are better than earlier ones against fixed targets,
  or the race is just going in circles.

The four project invariants all bind here, and are enforced at runtime rather
than only in tests:

1. adversary rows are featurized through ``graph.overlay`` -- the GA's own scoring
   path (``pool.py``);
2. every strategy is decoded by the money-conserving decoder (``pool.py``);
3. the threshold each round is ``threshold_for_budget`` on the clean holdout --
   a fixed *policy*, so the alert budget is identical in every round even though
   the numeric threshold is not;
4. the clean holdout is fingerprinted before the run and re-checked after: the
   run fails loudly if a single value of it changed.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from arena.adversary.fitness import EpisodeSampler, FitnessConfig
from arena.adversary.ga import GAConfig, distinct_top, evolve, naive_genome
from arena.adversary.genome import random_genome, to_dict
from arena.detector import evaluate, threshold_for_budget, time_split
from arena.detector.model import Detector, downsample_negatives, fit_detector
from arena.features import AccountGraph, featurize
from arena.loop.evaluation import score_pool, score_strategy
from arena.loop.pool import ADVERSARY_PATTERN_ID_BASE, StrategyPool


@dataclass(slots=True)
class ArenaSettings:
    """Closed-loop hyperparameters. Defaults mirror ``config/default.yaml``."""

    rounds: int = 5
    injection_cap: float = 0.20        # adversary rows <= this fraction of original positives
    top_k: int = 8                     # mutually-distinct strategies fed back per round
    instances_per_strategy: int = 6    # episodes each strategy is decoded on
    warm_start: bool = True            # seed each GA with the pool's recent strategies
    eval_episodes: int = 30            # fixed held-out episodes every pair is scored on
    n_random_probes: int = 10          # unseen random genomes, to test generalisation
    budget: float = 0.005              # the fixed alert budget (invariant 3)
    negative_ratio: float | None = 25.0
    seed: int = 20260917


@dataclass(slots=True)
class ArenaResult:
    """Everything the run measured. ``to_json`` is what ``reports/`` and the
    dashboard consume."""

    settings: ArenaSettings
    windows: dict[str, list[float]] = field(default_factory=dict)
    caps: dict[str, int] = field(default_factory=dict)
    detectors: list[dict] = field(default_factory=list)     # R + 1 entries
    adversaries: list[dict] = field(default_factory=list)   # R entries
    recall_matrix: list[list[float]] = field(default_factory=list)    # [detector][adversary]
    evasion_matrix: list[list[float]] = field(default_factory=list)
    episode: dict = field(default_factory=dict)
    holdout_fingerprint: int = 0
    runtime_minutes: float = 0.0
    cpu_minutes: float = 0.0

    def to_json(self) -> dict:
        return {
            "phase": 3,
            "settings": asdict(self.settings),
            "episode": self.episode,
            "windows_hours": self.windows,
            "row_caps": self.caps,
            "holdout_fingerprint": self.holdout_fingerprint,
            "runtime_minutes": round(self.runtime_minutes, 2),
            "cpu_minutes": round(self.cpu_minutes, 2),
            "detectors": self.detectors,
            "adversaries": self.adversaries,
            "crossplay": {"recall": self.recall_matrix, "evasion": self.evasion_matrix},
        }


def fingerprint(df: pd.DataFrame) -> int:
    """An order-independent content hash, for the invariant-4 runtime check."""
    return int(pd.util.hash_pandas_object(df, index=False).sum() % (2**63))


def _hours(ts, epoch: pd.Timestamp) -> float:
    return float((pd.Timestamp(ts) - epoch).total_seconds() / 3600.0)


def run_arena(
    ledger: pd.DataFrame,
    holdout: pd.DataFrame,
    detector0: Detector,
    settings: ArenaSettings,
    ga_cfg: GAConfig,
    fitness_cfg: FitnessConfig,
    *,
    amount_goal: float | tuple[float, float],
    deadline_hours: float,
    train_frac: float,
    valid_frac: float,
    lgb_params: dict | None = None,
    out_dir: str | Path | None = None,
    on_round: Callable[[ArenaResult], None] | None = None,
    progress: bool = True,
) -> ArenaResult:
    """Run the closed loop for ``settings.rounds`` rounds.

    Args:
        ledger: The full background ledger (benign + injected background typologies).
        holdout: The frozen clean holdout written in Phase 1. Never trained on,
            never augmented; verified unchanged at the end.
        detector0: The Phase 1 detector, ``D_0``.
        on_round: Called with the partial result after every round, so a long run
            can be watched (and a crash loses only the current round's report).
    """
    t_run, cpu_run = time.time(), time.process_time()
    say = print if progress else (lambda *a, **k: None)

    # ---- invariant 4: the holdout is clean now, and must be identical at the end
    if (holdout["source"].astype(str) != "background").any():
        raise ValueError("the holdout contains non-background rows -- it is not clean")
    if holdout["pattern_id"].dropna().ge(ADVERSARY_PATTERN_ID_BASE).any():
        raise ValueError("the holdout contains adversary pattern ids -- it is not clean")
    fp_before = fingerprint(holdout)

    split = time_split(ledger, train_frac, valid_frac)
    if len(split.holdout) != len(holdout):
        raise ValueError(
            f"the holdout ({len(holdout):,} rows) is not the chronological tail of this ledger "
            f"({len(split.holdout):,} rows) -- re-run scripts/train_detector.py on this ledger"
        )

    epoch = detector0.epoch
    window_hours = detector0.window_hours
    graph = AccountGraph.build(ledger, epoch=epoch)   # the background never changes: build once
    say(f"background graph frozen: {graph.n_accounts:,} accounts")

    # ---- features, built once. Adversary rows are appended per round; these never change.
    t0 = time.time()
    X_h = featurize(holdout, graph, window_hours)
    y_h = holdout["is_laundering"].to_numpy(dtype=int)
    typ_h = holdout["typology"]

    train_rows = split.train
    valid_rows = split.valid
    if settings.negative_ratio is not None:
        # Same sampling (and seed) as detector.model.train, so D_0 and every
        # later detector are trained on the *same* original rows.
        train_rows = downsample_negatives(train_rows, settings.negative_ratio, 0)
        valid_rows = downsample_negatives(valid_rows, settings.negative_ratio, 0)
    X_tr = featurize(train_rows, graph, window_hours)
    X_va = featurize(valid_rows, graph, window_hours)
    y_tr = train_rows["is_laundering"].to_numpy(dtype=int)
    y_va = valid_rows["is_laundering"].to_numpy(dtype=int)
    say(f"features built in {time.time() - t0:.0f}s  "
        f"(train {len(X_tr):,} rows / {int(y_tr.sum()):,} pos, "
        f"valid {len(X_va):,} / {int(y_va.sum()):,}, holdout {len(X_h):,} / {int(y_h.sum()):,})")

    # ---- time windows (hours since epoch): train / valid / held-out evaluation
    train_start = _hours(split.train["timestamp"].min(), epoch)
    train_end = _hours(split.train["timestamp"].max(), epoch)
    valid_end = _hours(split.valid["timestamp"].max(), epoch)
    ledger_end = _hours(ledger["timestamp"].max(), epoch)
    windows = {
        "train": (train_start, train_end),
        "valid": (train_end, valid_end),
        "eval": (valid_end, ledger_end),
    }
    cap_train = int(settings.injection_cap * int(y_tr.sum()))
    cap_valid = int(settings.injection_cap * int(y_va.sum()))

    result = ArenaResult(
        settings=settings,
        windows={k: [float(a), float(b)] for k, (a, b) in windows.items()},
        caps={"train_rows": cap_train, "valid_rows": cap_valid},
        episode={
            "amount_goal": list(amount_goal) if isinstance(amount_goal, tuple) else amount_goal,
            "amount_goal_kind": "log-uniform range" if isinstance(amount_goal, tuple) else "fixed",
            "deadline_hours": deadline_hours,
        },
        holdout_fingerprint=fp_before,
    )

    # ---- fixed evaluation set and probes: identical for every (detector, adversary) pair
    sampler = EpisodeSampler(ledger, epoch)
    eval_eps = sampler.sample(
        settings.eval_episodes, np.random.default_rng(settings.seed + 1),
        amount_goal, deadline_hours, start_range=(windows["eval"][0], windows["eval"][1] - deadline_hours),
    )
    naive = naive_genome()
    random_pool = [
        random_genome(np.random.default_rng(settings.seed + 100 + i))
        for i in range(settings.n_random_probes)
    ]
    fee_rate = fitness_cfg.fee_rate

    out = Path(out_dir) if out_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
        detector0.save(out / "detector_r0.lgb")

    detectors: list[Detector] = [detector0]
    champions: list[np.ndarray] = []
    pool = StrategyPool()
    injection: list[dict] = [{"n_instances": 0, "n_rows": 0, "rows_per_round": {}}]

    for r in range(settings.rounds + 1):
        D = detectors[r]

        # ---- invariant 3: the threshold is the fixed budget's quantile on the clean holdout
        scores_h = D.score_features(X_h)
        tau = threshold_for_budget(scores_h, settings.budget)
        hold = evaluate(y_h, scores_h, tau, typ_h, budget=settings.budget)
        hold.pop("roc_auc_note", None)
        result.detectors.append({
            "round": r,
            "tau": float(tau),
            "holdout": hold,
            "naive": score_strategy(naive, eval_eps, D, graph, tau, fitness_cfg, settings.seed + 7).as_dict(),
            "random": score_pool(random_pool, eval_eps, D, graph, tau, fitness_cfg, settings.seed + 7).as_dict(),
            "n_trees": int(D.booster.num_trees()),
            "injection": injection[r],
        })

        if r == settings.rounds:
            say(f"\nD_{r} (final, no adversary evolved against it): holdout recall@budget "
                f"{hold['recall_at_budget']:.3f}  PR-AUC {hold['pr_auc']:.3f}")
            break

        # ---- the fight: evolve A_r against D_r
        say(f"\n=== round {r} ===  D_{r} holdout recall@budget {hold['recall_at_budget']:.3f}  "
            f"PR-AUC {hold['pr_auc']:.3f}  tau {tau:.4f}")
        seeds = None
        if settings.warm_start and r > 0:
            seeds = pool.recent_genomes(ga_cfg.population // 2)
        t_ga, cpu_ga = time.time(), time.process_time()
        best, history, population = evolve(
            D, graph, tau, ledger, replace(ga_cfg, seed=ga_cfg.seed + r), fitness_cfg,
            amount_goal, deadline_hours, epoch, progress=progress, seed_genomes=seeds,
        )
        ga_minutes = (time.time() - t_ga) / 60.0
        ga_cpu_minutes = (time.process_time() - cpu_ga) / 60.0
        diverse = distinct_top(population, settings.top_k)
        champions.append(best)

        champ = score_strategy(best, eval_eps, D, graph, tau, fitness_cfg, settings.seed + 7)
        pool_score = (
            score_pool(diverse, eval_eps, D, graph, tau, fitness_cfg, settings.seed + 7)
            if diverse else champ
        )
        if not diverse:
            say(f"  WARNING: A_{r} found no strategy that completes the laundering goal "
                f"(best fitness {history[-1].best_fitness:.3f}) -- there is nothing to feed back")
        say(f"  A_{r} champion vs D_{r} on held-out episodes: detector recall "
            f"{champ.detector_recall:.3f}  evasion {champ.evasion:.3f}  success rate "
            f"{champ.success_rate:.2f}  (GA best fitness {history[-1].best_fitness:.3f}, "
            f"{ga_minutes:.1f} min)")

        # ---- the lesson: fold the pool back in and retrain from the ORIGINAL set
        pool.add_round(r, diverse)
        mat_tr = pool.materialize(
            sampler=sampler, graph=graph, window_hours=window_hours,
            rng=np.random.default_rng(settings.seed + 1000 * (r + 1) + 1),
            window=windows["train"], row_cap=cap_train,
            instances_per_strategy=settings.instances_per_strategy,
            amount_goal=amount_goal, deadline_hours=deadline_hours, fee_rate=fee_rate,
        )
        mat_va = pool.materialize(
            sampler=sampler, graph=graph, window_hours=window_hours,
            rng=np.random.default_rng(settings.seed + 1000 * (r + 1) + 2),
            window=windows["valid"], row_cap=cap_valid,
            instances_per_strategy=settings.instances_per_strategy,
            amount_goal=amount_goal, deadline_hours=deadline_hours, fee_rate=fee_rate,
            id_base=ADVERSARY_PATTERN_ID_BASE + 500_000,
        )
        X_aug = pd.concat([X_tr, mat_tr.X], ignore_index=True) if mat_tr.n_rows else X_tr
        y_aug = np.concatenate([y_tr, mat_tr.y]) if mat_tr.n_rows else y_tr
        Xv_aug = pd.concat([X_va, mat_va.X], ignore_index=True) if mat_va.n_rows else X_va
        yv_aug = np.concatenate([y_va, mat_va.y]) if mat_va.n_rows else y_va

        D_next = fit_detector(
            X_aug, y_aug, Xv_aug, yv_aug,
            window_hours=window_hours, epoch=epoch, params=lgb_params,
        )
        detectors.append(D_next)
        if out:
            D_next.save(out / f"detector_r{r + 1}.lgb")
        injection.append({
            "n_instances": int(mat_tr.n_instances),
            "n_rows": int(mat_tr.n_rows),
            "rows_per_round": {str(k): int(v) for k, v in mat_tr.rows_per_round.items()},
            "dropped_out_of_window": int(mat_tr.dropped_out_of_window),
            "dropped_over_cap": int(mat_tr.dropped_over_cap),
            "share_of_positives": float(mat_tr.n_rows / max(int(y_tr.sum()) + mat_tr.n_rows, 1)),
        })
        say(f"  fed back {mat_tr.n_instances} operations / {mat_tr.n_rows} rows "
            f"({mat_tr.n_rows / max(int(y_tr.sum()), 1):.0%} of original positives, cap "
            f"{settings.injection_cap:.0%}) -> D_{r + 1} ({D_next.booster.num_trees()} trees)")
        if mat_tr.dropped_out_of_window:
            say(f"  note: {mat_tr.dropped_out_of_window} decoded operation(s) ran past the training "
                f"period and were dropped whole rather than leak into the validation period")
        if mat_tr.n_rows == 0:
            say(f"  WARNING: no feedback reached D_{r + 1} -- it is the same model as D_{r} retrained")

        result.adversaries.append({
            "round": r,
            "ga": {
                "runtime_minutes": round(ga_minutes, 2),
                "cpu_minutes": round(ga_cpu_minutes, 2),
                "warm_start_seeds": len(seeds) if seeds else 0,
                "best_fitness": float(history[-1].best_fitness),
                "mean_fitness": float(history[-1].mean_fitness),
                "history": [
                    {"generation": h.generation, "best_fitness": h.best_fitness,
                     "mean_fitness": h.mean_fitness, "best_evasion": h.best_evasion,
                     "best_completion": h.best_completion, "diversity": h.diversity}
                    for h in history
                ],
            },
            "champion": to_dict(best),
            "strategies": [to_dict(g) for g in diverse],
            "champion_vs_own": champ.as_dict(),
            "pool_vs_own": pool_score.as_dict(),
        })
        if on_round:
            on_round(result)

    # ---- the cross-play matrix: every detector against every adversary's champion
    recall = np.zeros((len(detectors), len(champions)))
    evasion = np.zeros_like(recall)
    for i, D in enumerate(detectors):
        tau_i = result.detectors[i]["tau"]
        for j, g in enumerate(champions):
            s = score_strategy(g, eval_eps, D, graph, tau_i, fitness_cfg, settings.seed + 7)
            recall[i, j] = s.detector_recall
            evasion[i, j] = s.evasion
    result.recall_matrix = recall.tolist()
    result.evasion_matrix = evasion.tolist()

    # ---- invariant 4, verified rather than assumed
    if fingerprint(holdout) != fp_before:
        raise RuntimeError("INVARIANT 4 VIOLATED: the clean holdout changed during the run")

    result.runtime_minutes = (time.time() - t_run) / 60.0
    result.cpu_minutes = (time.process_time() - cpu_run) / 60.0
    if on_round:
        on_round(result)
    return result
