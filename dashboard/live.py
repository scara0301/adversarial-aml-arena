"""The interactive genome explorer's engine.

Loads the trained detector and the frozen background graph once per server
process (``st.cache_resource``), then lets the dashboard decode and score an
arbitrary genome on demand. This is the only place the dashboard touches the
live detector/adversary code instead of a report artifact -- everything else
in ``dashboard/`` just displays what ``scripts/train_detector.py`` and
``scripts/evolve_adversary.py`` already wrote to ``reports/``.

Scoring goes through ``evaluate_episode`` and ``Detector.score_ledger``, the
exact same functions the Phase 2 GA used for all 20,500 of its candidate
evaluations -- so a score shown here means the same thing it meant there.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

from arena.adversary.decoder import Episode
from arena.adversary.fitness import EpisodeSampler, FitnessConfig, evaluate_episode
from arena.config import Config
from arena.data.loader import load_ledger
from arena.detector import threshold_for_budget
from arena.detector.model import Detector
from arena.features import AccountGraph, featurize


@dataclass(slots=True)
class Engine:
    detector: Detector
    graph: AccountGraph
    tau: float
    episode: Episode
    fitness_cfg: FitnessConfig


def artifacts_available() -> bool:
    """Whether the files the engine needs actually exist on disk.

    ``data/`` and ``artifacts/`` are gitignored and regenerable, so a fresh
    clone of the repo won't have them -- this page must degrade gracefully
    rather than crash on import.
    """
    cfg = Config.load()
    return (
        cfg.path("paths.detector").exists()
        and cfg.path("paths.synthetic_ledger").exists()
        and cfg.path("paths.holdout_clean").exists()
    )


@st.cache_resource(
    show_spinner="Loading the trained detector and freezing the background graph "
    "(~10s the first time, cached after that)..."
)
def load_engine() -> Engine:
    cfg = Config.load()
    detector = Detector.load(cfg.path("paths.detector"))
    ledger = load_ledger("parquet", path=cfg.path("paths.synthetic_ledger"))
    graph = AccountGraph.build(ledger, epoch=detector.epoch)

    # Invariant 3: the same fixed alert-budget policy every other page uses.
    holdout = load_ledger("parquet", path=cfg.path("paths.holdout_clean"))
    budget = cfg.get("threshold.alert_budget", 0.005)
    tau = threshold_for_budget(
        detector.score_features(featurize(holdout, graph, detector.window_hours)), budget
    )

    # One fixed demo episode, drawn with a fixed seed, so every visitor's
    # slider tweaks are compared against the same laundering task.
    ep_cfg = cfg.section("adversary.episode")
    sampler = EpisodeSampler(ledger, detector.epoch)
    episode = sampler.sample(
        1,
        np.random.default_rng(7),
        ep_cfg.get("amount_goal", 250_000.0),
        ep_cfg.get("deadline_hours", 336.0),
    )[0]

    fit_cfg = cfg.section("adversary.fitness")
    fitness_cfg = FitnessConfig(
        alpha=fit_cfg.get("alpha", 1.0),
        beta=fit_cfg.get("beta", 1.0),
        gamma=fit_cfg.get("gamma", 0.1),
        lambda_time=fit_cfg.get("lambda_time", 0.3),
        completion_floor=fit_cfg.get("completion_floor", 0.95),
        fee_rate=cfg.get("adversary.constraints.fee_rate", 0.002),
    )

    return Engine(detector, graph, tau, episode, fitness_cfg)


def score_genome(
    genome: np.ndarray, seed: int = 0
) -> tuple[object, pd.DataFrame, np.ndarray, float]:
    """Decode + score one genome against the fixed demo episode.

    Returns ``(outcome, sub_ledger, scores, tau)``. ``scores`` are the
    detector's per-transaction probabilities, aligned to ``sub_ledger``'s
    rows, so the caller can show exactly which legs got flagged and which
    passed through.
    """
    engine = load_engine()
    rng = np.random.default_rng(seed)
    outcome = evaluate_episode(
        genome, engine.episode, engine.detector, engine.graph, engine.tau, rng, engine.fitness_cfg
    )
    sub = outcome.sub_ledger
    if sub.empty:
        return outcome, sub, np.zeros(0), engine.tau

    # Rescore explicitly rather than threading a score array out of
    # evaluate_episode: it's the same ~5ms overlay + featurize call, and this
    # way the fitness function's internals don't need to change shape just to
    # serve the dashboard.
    scores = engine.detector.score_ledger(sub, engine.graph.overlay(sub))
    return outcome, sub, scores, engine.tau
