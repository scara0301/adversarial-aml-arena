"""Detector, threshold policy and metrics tests.

The threshold tests cover invariant 3 and the holdout test covers invariant 4.
Both are about *comparability across arms-race rounds*, which is the property the
project's headline curve depends on and the easiest one to lose silently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from arena.detector import evaluate, threshold_for_budget, time_split, train
from arena.detector.metrics import (
    precision_at_threshold,
    recall_at_threshold,
    typology_recall,
)
from arena.detector.model import Detector, downsample_negatives
from arena.detector.threshold import flag_rate


# ----------------------------------------------------------------- threshold


def test_threshold_spends_exactly_the_alert_budget():
    """Invariant 3: the budget determines the operating point.

    If the realised flag rate drifted from the budget, recall would not be
    comparable between rounds and the arms-race curve would be meaningless.
    """
    rng = np.random.default_rng(0)
    scores = rng.random(200_000)
    for budget in (0.001, 0.005, 0.01, 0.05):
        tau = threshold_for_budget(scores, budget)
        assert flag_rate(scores, tau) == pytest.approx(budget, abs=0.0005)


def test_threshold_is_monotone_in_budget():
    scores = np.random.default_rng(1).random(50_000)
    assert threshold_for_budget(scores, 0.001) > threshold_for_budget(scores, 0.05)


def test_threshold_rejects_invalid_budgets():
    scores = np.random.default_rng(2).random(100)
    for bad in (0.0, 1.0, -0.1, 2.0):
        with pytest.raises(ValueError, match="budget must be"):
            threshold_for_budget(scores, bad)


def test_threshold_rejects_empty_scores():
    with pytest.raises(ValueError, match="empty score array"):
        threshold_for_budget(np.zeros(0), 0.005)


def test_threshold_handles_heavily_tied_scores():
    """A collapsed model produces mostly identical scores; this must not crash."""
    scores = np.concatenate([np.zeros(9_900), np.ones(100)])
    tau = threshold_for_budget(scores, 0.005)
    assert np.isfinite(tau)


# ------------------------------------------------------------------- metrics


def test_recall_and_precision_at_a_known_threshold():
    y = np.array([1, 1, 0, 0, 1, 0])
    s = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])
    # tau=0.75 flags the top two, both positive.
    assert recall_at_threshold(y, s, 0.75) == pytest.approx(2 / 3)
    assert precision_at_threshold(y, s, 0.75) == pytest.approx(1.0)


def test_recall_is_nan_without_positives():
    y = np.zeros(10, dtype=int)
    assert np.isnan(recall_at_threshold(y, np.random.default_rng(0).random(10), 0.5))


def test_typology_recall_handles_null_labels():
    """Benign rows carry a null typology; the mask must not swallow them.

    Comparing a nullable string column yields NA for benign rows, and an
    object-dtype mask cannot index a numpy array -- this was a real crash.
    """
    y = np.array([1, 1, 0, 1])
    s = np.array([0.9, 0.1, 0.5, 0.95])
    typ = pd.Series(["fan-out", "fan-out", None, "cycle"], dtype="string")
    out = typology_recall(y, s, typ, 0.5)
    assert out == {"cycle": 1.0, "fan-out": 0.5}


def test_evaluate_reports_the_honest_metric_set():
    rng = np.random.default_rng(3)
    y = (rng.random(20_000) < 0.002).astype(int)
    s = rng.random(20_000) * 0.5 + y * 0.4  # positives rank higher
    m = evaluate(y, s, threshold_for_budget(s, 0.005), budget=0.005)

    assert "accuracy" not in m, "accuracy must never be reported"
    assert m["recall_at_budget"] > 0.5
    assert "roc_auc_note" in m, "ROC-AUC must carry its caveat"
    assert m["flag_rate"] == pytest.approx(0.005, abs=0.001)


def test_evaluate_survives_a_single_class():
    y = np.zeros(1_000, dtype=int)
    s = np.random.default_rng(4).random(1_000)
    m = evaluate(y, s, 0.5, budget=0.005)
    assert np.isnan(m["pr_auc"]) and np.isnan(m["roc_auc"])


# --------------------------------------------------------------------- split


def test_time_split_is_chronological_and_disjoint(ledger):
    """Invariant 4: the holdout is the last slice in time.

    A random split would place transactions from the same laundering pattern,
    and the same account histories, on both sides of the boundary -- inflating
    every metric invisibly.
    """
    split = time_split(ledger, 0.6, 0.15)

    assert len(split.train) + len(split.valid) + len(split.holdout) == len(ledger)
    assert split.train["timestamp"].max() <= split.valid["timestamp"].min()
    assert split.valid["timestamp"].max() <= split.holdout["timestamp"].min()
    assert split.holdout["is_laundering"].sum() > 0, "holdout needs positives to score"


def test_time_split_requires_room_for_a_holdout(ledger):
    with pytest.raises(ValueError, match="leave room for a holdout"):
        time_split(ledger, 0.9, 0.2)


def test_holdout_contains_no_adversary_rows(ledger):
    """Invariant 4: nothing the adversary produced may reach the holdout."""
    split = time_split(ledger, 0.6, 0.15)
    assert (split.holdout["source"].astype(str) == "background").all()


# -------------------------------------------------------------- downsampling


def test_downsampling_keeps_every_positive(ledger):
    out = downsample_negatives(ledger, ratio=5.0, seed=0)
    assert out["is_laundering"].sum() == ledger["is_laundering"].sum()
    n_pos = int(out["is_laundering"].sum())
    assert (out["is_laundering"] == 0).sum() <= n_pos * 5


def test_downsampling_preserves_time_order(ledger):
    out = downsample_negatives(ledger, ratio=5.0, seed=0)
    assert out["timestamp"].is_monotonic_increasing


def test_downsampling_is_a_noop_when_already_balanced(ledger):
    out = downsample_negatives(ledger, ratio=10_000.0, seed=0)
    assert len(out) == len(ledger)


# ------------------------------------------------------- train and round-trip


def test_trained_detector_ranks_laundering_above_benign(ledger, graph):
    """A weak but real signal check: the model must beat chance.

    Deliberately loose. The fixture ledger is small, and a tight bound here
    would make the suite flaky without testing anything the Phase 1 metrics
    do not already measure.
    """
    split = time_split(ledger, 0.6, 0.15)
    detector, _ = train(split, graph, params={"n_estimators": 60}, negative_ratio=10.0)

    from arena.features import featurize

    scores = detector.score_features(featurize(split.holdout, graph))
    y = split.holdout["is_laundering"].to_numpy(dtype=int)
    if y.sum() == 0:
        pytest.skip("fixture holdout has no positives")
    assert scores[y == 1].mean() > scores[y == 0].mean()


def test_detector_round_trips_through_disk(ledger, graph, tmp_path):
    """Feature order and the epoch must survive save/load.

    The GA scores candidates thousands of times against a loaded detector; a
    silent feature-order mismatch would produce plausible-looking nonsense
    rather than an error.
    """
    split = time_split(ledger, 0.6, 0.15)
    detector, _ = train(split, graph, params={"n_estimators": 30}, negative_ratio=10.0)

    path = tmp_path / "d.lgb"
    detector.save(path)
    loaded = Detector.load(path)

    assert loaded.features == detector.features
    assert loaded.window_hours == detector.window_hours
    assert loaded.epoch == detector.epoch

    from arena.features import featurize

    X = featurize(split.holdout.head(200), graph)
    np.testing.assert_allclose(loaded.score_features(X), detector.score_features(X))


def test_score_features_tolerates_column_reordering(ledger, graph):
    """Reindexing must protect against a caller shuffling the columns."""
    split = time_split(ledger, 0.6, 0.15)
    detector, _ = train(split, graph, params={"n_estimators": 30}, negative_ratio=10.0)

    from arena.features import featurize

    X = featurize(split.holdout.head(100), graph)
    shuffled = X.loc[:, list(reversed(X.columns))]
    np.testing.assert_allclose(detector.score_features(X), detector.score_features(shuffled))
