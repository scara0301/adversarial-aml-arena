"""Feature pipeline tests, including the invariant-1 equivalence check.

``test_batch_and_overlay_agree`` is the most important test in the project. If it
fails, the adversary is being scored by a different function than the one that
produced the detector's training data, and every evasion number the project
reports is meaningless. Everything else here is ordinary correctness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from arena.features import AccountGraph, featurize
from arena.features.account_graph import to_hours
from arena.schema import REPORTING_THRESHOLD


def test_featurize_shape_and_cleanliness(ledger, graph):
    X = featurize(ledger.head(500), graph)
    assert len(X) == 500
    assert X.shape[1] > 60, "expected the full three-family feature set"
    assert not X.isna().any().any(), "features must never be NaN"
    assert np.isfinite(X.to_numpy()).all(), "features must never be inf"
    assert list(X.columns) == sorted(X.columns), "column order must be stable"


def test_column_order_is_independent_of_input(ledger, graph):
    a = featurize(ledger.head(50), graph)
    b = featurize(ledger.tail(50), graph)
    assert list(a.columns) == list(b.columns)


def test_batch_and_overlay_agree(ledger):
    """Invariant 1: one featurize, two ways of supplying the graph.

    Path A builds the graph over the whole ledger and featurizes a slice.
    Path B freezes the background, overlays the same slice, and featurizes it.

    These must agree exactly. Path A is how training data is built; path B is how
    the GA scores its candidates. If they diverge, the adversary is effectively
    attacking a different detector than the one that was trained, and the
    arms-race result is an artefact of the mismatch.
    """
    last_pid = ledger.loc[ledger["is_laundering"] == 1, "pattern_id"].max()
    # pattern_id is NA for benign rows, and NA != x yields NA, which would
    # silently drop every benign row from the "background".
    is_sub = (ledger["pattern_id"] == last_pid).fillna(False)
    sub, background = ledger[is_sub], ledger[~is_sub]

    assert len(sub) >= 3, "fixture should contain a multi-leg pattern"
    assert len(background) > 1000, "background must be substantial"

    epoch = pd.Timestamp(ledger["timestamp"].min())
    batch = featurize(sub, AccountGraph.build(ledger, epoch=epoch))
    overlay = featurize(sub, AccountGraph.build(background, epoch=epoch).overlay(sub))

    assert list(batch.columns) == list(overlay.columns)
    np.testing.assert_allclose(
        batch.to_numpy(), overlay.to_numpy(), rtol=1e-9, atol=1e-9,
        err_msg="INVARIANT 1 VIOLATED: batch and overlay featurization disagree",
    )


def test_overlay_leaves_frozen_graph_unchanged(ledger):
    """Copy-on-write must not mutate the frozen snapshot.

    The frozen graph is reused across thousands of GA candidates. If an overlay
    leaked a mutation into it, candidate N would be scored against the residue of
    candidates 1..N-1, and fitness would depend on evaluation order.
    """
    epoch = pd.Timestamp(ledger["timestamp"].min())
    frozen = AccountGraph.build(ledger, epoch=epoch)
    sub = ledger.head(20)

    before = featurize(sub, frozen).to_numpy().copy()
    frozen.overlay(sub)
    frozen.overlay(ledger.tail(30))
    after = featurize(sub, frozen).to_numpy()

    np.testing.assert_array_equal(before, after)


def test_as_of_features_exclude_the_present(ledger, graph):
    """An account's degree must not count the transaction being scored.

    Counting it would make every feature partly a function of its own row, which
    is a subtle self-leak: the first transaction an account ever makes would
    report out-degree 1 instead of 0.
    """
    first_txn = ledger.sort_values("timestamp").iloc[[0]]
    X = featurize(first_txn, graph)
    assert X["snd_out_deg"].iloc[0] == 0.0
    assert X["snd_out_vol"].iloc[0] == 0.0


def test_threshold_features_track_the_reporting_limit(graph, ledger):
    sub = ledger.head(1).copy()
    sub["amount_paid"] = REPORTING_THRESHOLD * 0.95
    sub["amount_received"] = REPORTING_THRESHOLD * 0.95
    X = featurize(sub, graph)
    assert X["is_just_below_threshold"].iloc[0] == 1.0
    assert X["threshold_ratio"].iloc[0] == pytest.approx(0.95)

    sub["amount_paid"] = REPORTING_THRESHOLD * 1.5
    sub["amount_received"] = REPORTING_THRESHOLD * 1.5
    X = featurize(sub, graph)
    assert X["is_just_below_threshold"].iloc[0] == 0.0


def test_window_counts_are_monotone_in_window_size(ledger, graph):
    """A 7-day window can never contain fewer events than a 1-hour window."""
    X = featurize(ledger.tail(300), graph)
    assert (X["snd_out_cnt_168h"] >= X["snd_out_cnt_24h"] - 1e-9).all()
    assert (X["snd_out_cnt_24h"] >= X["snd_out_cnt_1h"] - 1e-9).all()
    assert (X["rcv_in_cnt_168h"] >= X["rcv_in_cnt_24h"] - 1e-9).all()


def test_to_hours_matches_wall_clock():
    epoch = pd.Timestamp("2026-01-01")
    # format="ISO8601" because pandas 3.0 infers one format from the first
    # element and then rejects the rest.
    ts = pd.Series(
        pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-01 06:30"], format="ISO8601")
    )
    np.testing.assert_allclose(to_hours(ts, epoch), [0.0, 24.0, 6.5])


def test_cycle_flag_fires_on_a_real_cycle(ledger):
    """A reciprocal pair must register as a 2-cycle."""
    epoch = pd.Timestamp("2026-01-01")
    pair = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-02", "2026-01-03"]),
            "from_bank": [1, 2],
            "from_account": ["A", "B"],
            "to_bank": [2, 1],
            "to_account": ["B", "A"],
            "amount_paid": [1000.0, 900.0],
            "payment_currency": ["US Dollar"] * 2,
            "amount_received": [1000.0, 900.0],
            "receiving_currency": ["US Dollar"] * 2,
            "payment_format": ["Wire"] * 2,
            "is_laundering": [0, 0],
            "typology": pd.array([None, None], dtype="string"),
            "pattern_id": pd.array([None, None], dtype="Int64"),
            "source": ["background"] * 2,
        }
    )
    from arena import schema

    pair = schema.coerce(pair)
    graph = AccountGraph.build(pair, epoch=epoch)
    X = featurize(pair, graph)
    # B -> A closes the loop opened by A -> B.
    assert X["cycle2"].iloc[1] == 1.0
