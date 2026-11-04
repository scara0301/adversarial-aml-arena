"""Closed-loop (Phase 3) tests.

The pool tests pin the guards against the failure modes retraining on adversary
output invites (collapse, forgetting, time leakage, GA artefacts). The
end-to-end test runs the whole loop on a tiny ledger and checks the project
invariants *as they bind in Phase 3*: the holdout is untouched, the alert budget
is identical every round, and the feedback never exceeds its cap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from arena import schema
from arena.adversary.fitness import EpisodeSampler, FitnessConfig
from arena.adversary.ga import GAConfig, evolve, naive_genome
from arena.adversary.genome import random_genome
from arena.detector import time_split, train
from arena.features import AccountGraph, featurize
from arena.features.account_graph import to_hours
from arena.loop import ArenaSettings, StrategyPool, fingerprint, run_arena
from arena.loop.evaluation import EpisodeOutcomes, bootstrap_ci
from arena.loop.reevaluate import reevaluate
from arena.loop.pool import ADVERSARY_PATTERN_ID_BASE

WINDOWS = (1.0, 24.0, 168.0)
DEADLINE = 24.0


@pytest.fixture(scope="module")
def sampler(ledger, graph) -> EpisodeSampler:
    return EpisodeSampler(ledger, graph.epoch)


@pytest.fixture(scope="module")
def full_window(ledger, graph) -> tuple[float, float]:
    end = (pd.Timestamp(ledger["timestamp"].max()) - graph.epoch).total_seconds() / 3600.0
    return (0.0, end)


def _pool(rounds: int, per_round: int = 3) -> StrategyPool:
    pool = StrategyPool()
    for r in range(rounds):
        pool.add_round(r, [random_genome(np.random.default_rng(100 * r + i)) for i in range(per_round)])
    return pool


def _materialize(pool, sampler, graph, window, cap, seed=0, per_strategy=4):
    return pool.materialize(
        sampler=sampler, graph=graph, window_hours=WINDOWS, rng=np.random.default_rng(seed),
        window=window, row_cap=cap, instances_per_strategy=per_strategy,
        amount_goal=250_000.0, deadline_hours=DEADLINE, fee_rate=0.002,
    )


# ------------------------------------------------------------------- the pool


def test_row_cap_is_never_exceeded(ledger, graph, sampler, full_window):
    """The guard against detector collapse: adversary rows are capped."""
    for cap in (10, 40, 150):
        m = _materialize(_small_pool(2), sampler, graph, full_window, cap)
        assert 0 < m.n_rows <= cap
        assert len(m.X) == m.n_rows


def test_a_zero_cap_or_empty_pool_yields_nothing(ledger, graph, sampler, full_window):
    assert _materialize(_small_pool(2), sampler, graph, full_window, 0).n_rows == 0
    assert _materialize(StrategyPool(), sampler, graph, full_window, 100).n_rows == 0


def _small_pool(rounds: int) -> StrategyPool:
    """Strategies that decode to a handful of rows each, varying by round."""
    from arena.adversary.genome import GENE_INDEX

    pool = StrategyPool()
    for r in range(rounds):
        genomes = []
        for k in range(3):
            g = naive_genome().copy()
            g[GENE_INDEX["n_splits"]] = 2 + k + r
            genomes.append(g)
        pool.add_round(r, genomes)
    return pool


def test_cap_is_shared_across_rounds(ledger, graph, sampler, full_window):
    """The guard against forgetting old adversaries.

    If the pool let the newest round fill the cap, a detector could buy the new
    strategy by dropping the old one. Each round gets an equal slice of the cap,
    counted in rows.
    """
    m = _materialize(_small_pool(3), sampler, graph, full_window, cap=60, per_strategy=6)
    shares = [m.rows_per_round[r] for r in sorted(m.rows_per_round)]
    assert min(shares) > 0, f"a round was starved of its slice: {shares}"
    biggest_instance = int(m.rows.groupby("pattern_id").size().max())
    assert max(shares) - min(shares) <= 2 * biggest_instance, f"unequal shares: {shares}"


def test_a_round_of_huge_operations_cannot_starve_the_others(ledger, graph, sampler, full_window):
    """Regression guard. An earlier version took one *operation* per round per
    pass -- equal only if operations are the same size. Decoded strategies run
    from 2 to over 100 rows, so one round's large operations filled the cap and a
    later round got 0 rows: the forgetting guard failing silently."""
    from arena.adversary.genome import GENE_INDEX

    big = naive_genome().copy()
    big[GENE_INDEX["threshold_hug"]] = 1.0       # ~26 tranches per operation
    big[GENE_INDEX["n_splits"]] = 12
    pool = StrategyPool()
    pool.add_round(0, [big, big, big])
    pool.add_round(1, _small_pool(1).genomes(0))
    m = _materialize(pool, sampler, graph, full_window, cap=120, per_strategy=6)
    assert m.rows_per_round[1] > 0, f"round 1 was starved: {m.rows_per_round}"


def test_rows_are_labelled_laundering_from_the_adversary(ledger, graph, sampler, full_window):
    m = _materialize(_small_pool(2), sampler, graph, full_window, 120)
    assert m.n_rows > 0, "vacuous: nothing was materialised"
    assert (m.rows["is_laundering"] == 1).all()
    assert (m.rows["source"].astype(str) == "adversary").all()
    assert m.rows["typology"].notna().all()
    schema.validate(m.rows)
    ids = m.rows["pattern_id"].astype(int)
    assert (ids >= ADVERSARY_PATTERN_ID_BASE).all(), "adversary ids must not collide with background ids"
    assert ids.nunique() == m.n_instances


def test_rows_stay_inside_the_window(ledger, graph, sampler):
    """The guard against time leakage: training-period adversary rows are decoded
    only from operations that start *and finish* inside the training period."""
    window = (200.0, 700.0)
    m = _materialize(_small_pool(2), sampler, graph, window, 200)
    assert m.n_rows > 0, "vacuous: nothing was materialised"
    hours = to_hours(m.rows["timestamp"], graph.epoch)
    assert hours.min() >= window[0] - 1e-6
    assert hours.max() <= window[1] + 1e-6


def test_operations_that_overrun_the_window_are_dropped_whole(ledger, graph, sampler):
    """A random genome decodes to an operation that runs for hundreds of hours.
    Kept, it would put training-period rows after the training cutoff; cut, it
    would be half a laundering pattern with a positive label. Dropped whole."""
    m = _materialize(_pool(2), sampler, graph, (200.0, 700.0), 200)
    assert m.dropped_out_of_window > 0
    if m.n_rows:
        assert to_hours(m.rows["timestamp"], graph.epoch).max() <= 700.0 + 1e-6


def test_a_window_shorter_than_the_deadline_is_refused(ledger, graph, sampler):
    with pytest.raises(ValueError, match="empty episode window"):
        _materialize(_pool(1), sampler, graph, (100.0, 100.0 + DEADLINE / 2), 50)


def test_pool_features_come_from_the_overlay_path(ledger, graph, sampler, full_window):
    """Invariant 1, as it binds in Phase 3: adversary rows reach the detector
    through the exact path the GA scored them by -- one instance at a time,
    against the frozen graph plus that instance's overlay."""
    m = _materialize(_small_pool(1), sampler, graph, full_window, 120)
    assert m.n_rows > 0, "vacuous: nothing was materialised"
    first_id = m.rows["pattern_id"].iloc[0]
    inst = m.rows[m.rows["pattern_id"] == first_id]
    expected = featurize(inst, graph.overlay(inst), WINDOWS)
    got = m.X.iloc[: len(inst)].reset_index(drop=True)
    np.testing.assert_allclose(got.to_numpy(), expected.reset_index(drop=True).to_numpy(), rtol=1e-9, atol=1e-9)


def test_materialisation_is_reproducible(ledger, graph, sampler, full_window):
    a = _materialize(_small_pool(2), sampler, graph, full_window, 100, seed=5)
    b = _materialize(_small_pool(2), sampler, graph, full_window, 100, seed=5)
    assert a.n_rows > 0, "vacuous: nothing was materialised"
    pd.testing.assert_frame_equal(a.X, b.X)


def test_recent_genomes_are_newest_first():
    pool = _pool(3, per_round=2)
    recent = pool.recent_genomes(3)
    assert len(recent) == 3
    np.testing.assert_array_equal(recent[0], pool.genomes(2)[0])
    np.testing.assert_array_equal(recent[2], pool.genomes(1)[0])


# ------------------------------------------------------------- episode goals


def test_goal_range_draws_varied_cent_precise_goals(sampler):
    eps = sampler.sample(300, np.random.default_rng(0), (50_000.0, 400_000.0), DEADLINE)
    goals = np.array([e.amount_goal for e in eps])
    assert goals.min() >= 50_000.0 and goals.max() <= 400_000.0
    assert len(set(goals)) > 250, "goals must vary from episode to episode"
    np.testing.assert_allclose(goals, np.round(goals, 2))
    # log-uniform: the median sits near the geometric mean (~$141k), not the arithmetic ($225k)
    assert 115_000 < np.median(goals) < 170_000


def test_a_fixed_goal_is_still_supported(sampler):
    eps = sampler.sample(5, np.random.default_rng(0), 250_000.0, DEADLINE)
    assert {e.amount_goal for e in eps} == {250_000.0}


def test_an_invalid_goal_range_is_refused(sampler):
    with pytest.raises(ValueError, match="amount_goal range"):
        sampler.sample(1, np.random.default_rng(0), (400_000.0, 50_000.0), DEADLINE)


def test_a_single_tranche_strategy_is_only_round_when_the_goal_is(ledger, graph, sampler):
    """Regression guard for the exploit the first closed-loop run found.

    With one tranche the decoder sends exactly the goal. The generator's laundering
    is never a round number but 21.6% of benign traffic is, so a detector learns
    "round means benign" -- and a fixed $250,000.00 goal made every one-tranche
    operation round. The GA found that in round 1 (detector recall 0.000). Drawing
    each episode's goal to the cent removes the free evasion.
    """
    from arena.adversary.decoder import decode
    from arena.adversary.genome import GENE_INDEX

    g = naive_genome().copy()
    g[GENE_INDEX["n_splits"]] = 1
    g[GENE_INDEX["n_hops"]] = 1
    g[GENE_INDEX["threshold_hug"]] = 0.0

    fixed = sampler.sample(1, np.random.default_rng(0), 250_000.0, DEADLINE)[0]
    sub = decode(g, fixed, np.random.default_rng(1), 0.002)
    assert len(sub) == 1 and sub["amount_paid"].iloc[0] % 100 == 0, "a fixed goal gives a round amount"

    varied = sampler.sample(40, np.random.default_rng(0), (50_000.0, 400_000.0), DEADLINE)
    amounts = [decode(g, e, np.random.default_rng(1), 0.002)["amount_paid"].iloc[0] for e in varied]
    round_share = np.mean([a % 100 == 0 for a in amounts])
    assert round_share < 0.1, f"{round_share:.0%} of one-tranche amounts are still round"


# --------------------------------------------------------------------- the GA


class _StubDetector:
    window_hours = WINDOWS
    features: list[str] = []

    def score_ledger(self, ledger, graph):  # noqa: ARG002
        return np.zeros(len(ledger))


def test_seed_genomes_enter_the_initial_population(ledger, graph):
    """Warm start: last round's strategies are carried into the new fight."""
    seed = naive_genome()
    cfg = GAConfig(population=10, generations=0, episodes_per_eval=1, elitism=1, seed=1)
    _, _, population = evolve(
        _StubDetector(), graph, 0.5, ledger, cfg, FitnessConfig(), 250_000.0, DEADLINE,
        graph.epoch, progress=False, seed_genomes=[seed],
    )
    assert any(np.allclose(np.asarray(ind), seed) for ind in population)


# ------------------------------------------------------------ the measurement


def _coin_flips(n: int, rng) -> EpisodeOutcomes:
    """A strategy that sends one transaction per episode: flagged or not, nothing between."""
    flagged = (rng.random(n) < 0.5).astype(float)
    return EpisodeOutcomes(
        n_flagged=flagged, n_transactions=np.ones(n), evasion=1.0 - flagged,
        completion=np.ones(n), fitness=1.0 - flagged, success=(1.0 - flagged) > 0.5,
    )


def test_bootstrap_interval_brackets_the_estimate_and_narrows_with_more_episodes():
    """Why the arms race is re-measured on more episodes. A one-transaction strategy
    is a coin flip per episode, so 30 episodes leave a standard error near 0.09 --
    wider than the round-to-round differences one would want to read into."""
    rng = np.random.default_rng(0)
    small, large = _coin_flips(30, rng), _coin_flips(400, rng)
    for o in (small, large):
        est = float(o.n_flagged.sum() / o.n_transactions.sum())
        lo, hi = bootstrap_ci(o, np.random.default_rng(1), n_boot=500)["detector_recall"]
        assert lo <= est <= hi
    w_small = np.subtract(*bootstrap_ci(small, np.random.default_rng(1), 500)["detector_recall"][::-1])
    w_large = np.subtract(*bootstrap_ci(large, np.random.default_rng(1), 500)["detector_recall"][::-1])
    assert w_small > 0.25, f"30 coin-flip episodes should be wide, got {w_small:.2f}"
    assert w_large < 0.6 * w_small


# ------------------------------------------------------------- the whole loop


def test_closed_loop_end_to_end(tmp_path):
    """Two rounds on a small ledger, with the invariants checked as they bind here.

    Builds its own 120-day ledger rather than using the shared 45-day fixture:
    the loop draws training, validation and evaluation episodes from three
    *separate* periods, and each must be long enough to hold an operation that
    starts and finishes inside it. In 45 days the validation period is under a
    week, so no episode can fit.
    """
    from arena.data.synthetic import generate

    ledger = generate(accounts=800, rows=12_000, laundering_rate=0.03, days=120, seed=7)
    graph = AccountGraph.build(ledger)
    split = time_split(ledger, 0.6, 0.15)
    holdout = split.holdout.copy()
    holdout_before = holdout.copy(deep=True)

    d0, _ = train(split, graph, params={"n_estimators": 40}, negative_ratio=10.0)
    settings = ArenaSettings(
        rounds=2, injection_cap=0.25, top_k=3, instances_per_strategy=3,
        eval_episodes=4, n_random_probes=3, negative_ratio=10.0, seed=7,
    )
    ga = GAConfig(population=24, generations=3, episodes_per_eval=2, elitism=2, seed=11)

    result = run_arena(
        ledger, holdout, d0, settings, ga, FitnessConfig(),
        amount_goal=(50_000.0, 300_000.0), deadline_hours=120.0, train_frac=0.6, valid_frac=0.15,
        lgb_params={"n_estimators": 40}, progress=False, out_dir=tmp_path,
    )
    res = result.to_json()

    # shape: R + 1 detectors, R adversaries, a full cross-play matrix
    assert len(res["detectors"]) == 3
    assert len(res["adversaries"]) == 2
    M = np.array(res["crossplay"]["recall"])
    assert M.shape == (3, 2)
    assert ((M >= 0) & (M <= 1)).all()

    # invariant 4: the holdout is exactly what it was
    pd.testing.assert_frame_equal(holdout, holdout_before)
    assert fingerprint(holdout) == res["holdout_fingerprint"]

    # invariant 3: the same alert *budget* every round, whatever the threshold
    for d in res["detectors"]:
        assert d["holdout"]["flag_rate"] == pytest.approx(settings.budget, abs=0.003)

    # the feedback cap is honoured, and the first detector received no feedback
    cap = res["row_caps"]["train_rows"]
    assert res["detectors"][0]["injection"]["n_rows"] == 0
    for d in res["detectors"][1:]:
        assert 0 < d["injection"]["n_rows"] <= cap

    # warm start: round 0 has nothing to seed from, round 1 does
    assert res["adversaries"][0]["ga"]["warm_start_seeds"] == 0
    assert res["adversaries"][1]["ga"]["warm_start_seeds"] > 0

    # every adversary reports a champion and a diverse pool
    for a in res["adversaries"]:
        assert 1 <= len(a["strategies"]) <= settings.top_k
        assert 0.0 <= a["champion_vs_own"]["detector_recall"] <= 1.0


    # re-measuring from the saved detectors and champions: same shapes, valid intervals
    from arena.detector.model import Detector

    detectors = [Detector.load(tmp_path / f"detector_r{i}.lgb") for i in range(3)]
    new = reevaluate(res, ledger, detectors, FitnessConfig(), n_episodes=6, n_boot=40)
    R = np.array(new["crossplay"]["recall"])
    lo, hi = np.array(new["crossplay"]["recall_ci_low"]), np.array(new["crossplay"]["recall_ci_high"])
    assert R.shape == (3, 2)
    assert (lo <= R + 1e-9).all() and (R <= hi + 1e-9).all()
    assert new["evaluation"]["episodes"] == 6 and new["settings"]["eval_episodes"] == 6
    for a in new["adversaries"]:
        assert set(a["champion_vs_own"]["ci"]) == {"detector_recall", "evasion"}
    # the search results and holdout metrics are untouched -- only the measurement changed
    assert new["detectors"][1]["holdout"] == res["detectors"][1]["holdout"]
    assert new["adversaries"][0]["champion"] == res["adversaries"][0]["champion"]
