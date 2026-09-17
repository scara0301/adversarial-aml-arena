"""The single shared feature pipeline.

**Invariant 1.** There is exactly one ``featurize()``. Offline detector training
calls it against a graph built from the full ledger; the GA adversary calls it
against an overlay of its candidate sub-ledger on a frozen background. The
adversary therefore never touches a feature vector -- it produces transactions
and they are featurized by the same code that produced the training data. An
adversary allowed to edit features directly would be cheating, and the project's
central claim would be void.

Three feature families:

**Per-transaction (stateless).** Amount, currency, format, time-of-day, and
crucially *threshold proximity*. The $10,000 reporting threshold is the single
most important number in the project: proximity to it is what makes structuring
detectable, and therefore the signal the adversary must learn to fight.

**Account aggregates (as-of).** Degree, unique counterparties, volume, flow
ratio, age, reciprocity, inter-arrival. Computed for both sender and receiver,
because a laundering leg is suspicious partly because of who is on the other end.

**Relational / typology.** Rolling window bursts over 1h / 24h / 7d, a
gather-scatter score, and closed-cycle flags for lengths 2-4. These are what let
the detector recognise the five typologies by shape rather than by amount alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from arena.features import windows as W
from arena.features.account_graph import AccountGraph, to_hours
from arena.schema import CURRENCIES, PAYMENT_FORMATS, REPORTING_THRESHOLD

#: Rolling windows in hours: one hour, one day, one week.
DEFAULT_WINDOWS = (1.0, 24.0, 168.0)

_FORMAT_CODE = {f: i for i, f in enumerate(PAYMENT_FORMATS)}
_CURRENCY_CODE = {c: i for i, c in enumerate(CURRENCIES)}

#: Formats ordered by how much scrutiny they attract, normalised to [0, 1].
#: The adversary's ``format_mix`` gene slides along this axis, so the detector
#: needs it as an explicit feature rather than only as a raw category.
_FORMAT_SCRUTINY = {
    "ACH": 0.15,
    "Cheque": 0.25,
    "Credit Card": 0.30,
    "Reinvestment": 0.35,
    "Wire": 0.70,
    "Cash": 0.85,
    "Bitcoin": 1.00,
}


def _transaction_features(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Stateless per-row features. Pure functions of the transaction itself."""
    amt = df["amount_paid"].to_numpy(dtype=np.float64)
    recv = df["amount_received"].to_numpy(dtype=np.float64)
    ts = pd.to_datetime(df["timestamp"])

    hour = ts.dt.hour.to_numpy(dtype=np.float64)
    dow = ts.dt.dayofweek.to_numpy(dtype=np.float64)

    fmt = df["payment_format"].astype("string").to_numpy(dtype=object)
    pay_cur = df["payment_currency"].astype("string").to_numpy(dtype=object)
    recv_cur = df["receiving_currency"].astype("string").to_numpy(dtype=object)

    return {
        "amount_log": np.log1p(amt),
        # Structuring leaves fingerprints in the low-order digits.
        "amount_mod_1000": np.mod(amt, 1000.0),
        "amount_mod_100": np.mod(amt, 100.0),
        "is_round_100": (np.mod(amt, 100.0) < 1e-6).astype(np.float64),
        # Threshold proximity: the core structuring signal.
        "threshold_ratio": amt / REPORTING_THRESHOLD,
        "is_just_below_threshold": (
            (amt >= 0.85 * REPORTING_THRESHOLD) & (amt < REPORTING_THRESHOLD)
        ).astype(np.float64),
        "threshold_gap": np.abs(amt - REPORTING_THRESHOLD) / REPORTING_THRESHOLD,
        "hour": hour,
        "day_of_week": dow,
        "is_business_hours": ((hour >= 9) & (hour < 17) & (dow < 5)).astype(np.float64),
        "is_weekend": (dow >= 5).astype(np.float64),
        "cross_bank": (
            df["from_bank"].to_numpy() != df["to_bank"].to_numpy()
        ).astype(np.float64),
        "cross_currency": (pay_cur != recv_cur).astype(np.float64),
        # Value lost in transit. Cycles skim at each hop, so this is small but
        # systematically non-zero for layering.
        "fx_loss": 1.0 - np.divide(recv, np.maximum(amt, 1e-9)),
        "format_code": np.array([_FORMAT_CODE.get(f, -1) for f in fmt], dtype=np.float64),
        "format_scrutiny": np.array(
            [_FORMAT_SCRUTINY.get(f, 0.5) for f in fmt], dtype=np.float64
        ),
        "currency_code": np.array([_CURRENCY_CODE.get(c, -1) for c in pay_cur], dtype=np.float64),
    }


def _side_features(
    accounts: np.ndarray,
    counterparties: np.ndarray,
    hours: np.ndarray,
    graph: AccountGraph,
    prefix: str,
    window_hours: tuple[float, ...],
) -> dict[str, np.ndarray]:
    """As-of account features for one side of the transaction.

    Rows are grouped by account so each account's arrays are queried once,
    vectorised over all its query times. This is what keeps the batch path
    tractable (a few tens of thousands of groups) while leaving the overlay path
    -- a handful of groups -- essentially free. Both run this same function.
    """
    n = len(accounts)
    names = [
        "out_deg",
        "in_deg",
        "out_vol",
        "in_vol",
        "out_uniq_cp",
        "in_uniq_cp",
        "flow_ratio",
        "mean_out_amt",
        "age_h",
        "recip_ratio",
        "since_prev_h",
        "pair_reps",
    ]
    for w in window_hours:
        tag = f"{int(w)}h"
        names += [f"out_cnt_{tag}", f"out_vol_{tag}", f"in_cnt_{tag}", f"in_vol_{tag}"]
    names += ["gather_scatter_24h", "burst_ratio"]

    out: dict[str, np.ndarray] = {f"{prefix}_{k}": np.zeros(n, dtype=np.float64) for k in names}

    order = np.argsort(accounts.astype(str), kind="stable")
    sorted_accounts = accounts[order]
    boundaries = np.flatnonzero(
        np.r_[True, sorted_accounts[1:] != sorted_accounts[:-1], True]
    )

    for b in range(len(boundaries) - 1):
        rows = order[boundaries[b] : boundaries[b + 1]]
        acct = accounts[rows[0]]
        state = graph.state(acct)
        q = hours[rows]
        cps = counterparties[rows]

        out_cnt, out_vol = W.counts_and_volume(state, q, "out")
        in_cnt, in_vol = W.counts_and_volume(state, q, "in")

        out[f"{prefix}_out_deg"][rows] = out_cnt
        out[f"{prefix}_in_deg"][rows] = in_cnt
        out[f"{prefix}_out_vol"][rows] = out_vol
        out[f"{prefix}_in_vol"][rows] = in_vol
        out[f"{prefix}_out_uniq_cp"][rows] = W.unique_counterparties(state, q, "out")
        out[f"{prefix}_in_uniq_cp"][rows] = W.unique_counterparties(state, q, "in")
        # Pass-through accounts sit near 0.5: money in, money straight out.
        out[f"{prefix}_flow_ratio"][rows] = in_vol / np.maximum(in_vol + out_vol, 1.0)
        out[f"{prefix}_mean_out_amt"][rows] = out_vol / np.maximum(out_cnt, 1.0)
        out[f"{prefix}_age_h"][rows] = np.where(
            np.isfinite(state.first_seen), q - state.first_seen, 0.0
        )
        out[f"{prefix}_recip_ratio"][rows] = W.reciprocity(state, q)
        out[f"{prefix}_since_prev_h"][rows] = W.since_previous_event(state, q)
        out[f"{prefix}_pair_reps"][rows] = W.pair_repetition(state, q, cps)

        w_out_cnt: dict[float, np.ndarray] = {}
        w_in_cnt: dict[float, np.ndarray] = {}
        for w in window_hours:
            tag = f"{int(w)}h"
            oc, ov = W.window_counts(state, q, "out", w)
            ic, iv = W.window_counts(state, q, "in", w)
            out[f"{prefix}_out_cnt_{tag}"][rows] = oc
            out[f"{prefix}_out_vol_{tag}"][rows] = ov
            out[f"{prefix}_in_cnt_{tag}"][rows] = ic
            out[f"{prefix}_in_vol_{tag}"][rows] = iv
            w_out_cnt[w] = oc
            w_in_cnt[w] = ic

        # Gather-scatter: inbound burst *and* outbound burst in the same day.
        # min() rather than sum() because the typology needs both legs.
        if 24.0 in w_out_cnt:
            out[f"{prefix}_gather_scatter_24h"][rows] = np.minimum(w_in_cnt[24.0], w_out_cnt[24.0])
        # Burstiness: how much of a day's activity landed in the last hour.
        if 1.0 in w_out_cnt and 24.0 in w_out_cnt:
            out[f"{prefix}_burst_ratio"][rows] = w_out_cnt[1.0] / np.maximum(w_out_cnt[24.0], 1.0)

    return out


def featurize(
    ledger: pd.DataFrame,
    graph: AccountGraph,
    window_hours: tuple[float, ...] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """Turn transactions into the detector's feature matrix.

    Args:
        ledger: Transactions in the canonical schema. In the batch path this is
            the whole training ledger; in the GA path it is one candidate
            sub-ledger of a few dozen rows.
        graph: Account state. ``AccountGraph.build(full_ledger)`` for batch, or
            ``frozen.overlay(sub_ledger)`` for the GA. Both are the same type
            and go through the same code below -- that is invariant 1.
        window_hours: Rolling window sizes.

    Returns:
        A float feature frame with one row per input transaction, in stable
        column order. The index matches ``ledger``'s.
    """
    if ledger.empty:
        return pd.DataFrame(index=ledger.index)

    hours = to_hours(ledger["timestamp"], graph.epoch)
    from_acc = ledger["from_account"].astype("string").to_numpy(dtype=object)
    to_acc = ledger["to_account"].astype("string").to_numpy(dtype=object)

    cols: dict[str, np.ndarray] = {}
    cols.update(_transaction_features(ledger))
    cols.update(_side_features(from_acc, to_acc, hours, graph, "snd", window_hours))
    cols.update(_side_features(to_acc, from_acc, hours, graph, "rcv", window_hours))
    cols.update({k: v.astype(np.float64) for k, v in graph.cycles.cycle_flags(from_acc, to_acc).items()})

    # Relative amount: a $9,000 transfer is unremarkable from an account that
    # routinely moves $100k and glaring from one that usually moves $200.
    amt = ledger["amount_paid"].to_numpy(dtype=np.float64)
    cols["amount_vs_snd_mean"] = amt / np.maximum(cols["snd_mean_out_amt"], 1.0)
    cols["amount_vs_rcv_mean"] = amt / np.maximum(cols["rcv_mean_out_amt"], 1.0)

    out = pd.DataFrame(cols, index=ledger.index)
    return out.loc[:, sorted(out.columns)].astype(np.float64)


def feature_names(window_hours: tuple[float, ...] = DEFAULT_WINDOWS) -> list[str]:
    """Stable feature column order, without needing a ledger to inspect.

    Derived by featurizing a tiny synthetic frame rather than by maintaining a
    hand-written list, so it can never drift out of sync with ``featurize``.
    """
    from arena.data.synthetic import generate

    tiny = generate(accounts=40, rows=200, laundering_rate=0.05, days=10, seed=0)
    graph = AccountGraph.build(tiny)
    return list(featurize(tiny.head(5), graph, window_hours).columns)
