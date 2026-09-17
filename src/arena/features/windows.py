"""As-of window queries against an :class:`~arena.features.account_graph.AccountState`.

All queries are vectorised over an array of query times for a *single* account.
That shape is deliberate: the batch path groups a large ledger by account and
calls these once per group, while the overlay path calls them for a handful of
accounts. Same functions, same results -- which is what makes invariant 1 hold
by construction rather than by a matching pair of implementations.
"""

from __future__ import annotations

import numpy as np

from arena.features.account_graph import AccountState


def _as_of_index(times: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Number of events strictly before each query time.

    ``side="left"`` excludes events at exactly the query time. That matters: a
    transaction must not count itself, nor its own siblings injected at the same
    instant, or the feature would peek at the very batch being scored.
    """
    if times.size == 0:
        return np.zeros(len(query), dtype=np.int64)
    return np.searchsorted(times, query, side="left")


def counts_and_volume(
    state: AccountState, query: np.ndarray, direction: str
) -> tuple[np.ndarray, np.ndarray]:
    """As-of event count and cumulative volume in one direction."""
    if direction == "out":
        times, cumvol = state.out_times, state.out_cumvol
    else:
        times, cumvol = state.in_times, state.in_cumvol
    idx = _as_of_index(times, query)
    return idx.astype(np.float64), cumvol[idx]


def unique_counterparties(state: AccountState, query: np.ndarray, direction: str) -> np.ndarray:
    """As-of distinct counterparty count.

    Exact, via the prefix sum of the per-event "first time seeing this
    counterparty" flags computed during the graph build walk. A searchsorted
    alone could not answer this.
    """
    if direction == "out":
        times, cumfirst = state.out_times, state.out_cumfirst
    else:
        times, cumfirst = state.in_times, state.in_cumfirst
    return cumfirst[_as_of_index(times, query)].astype(np.float64)


def reciprocity(state: AccountState, query: np.ndarray) -> np.ndarray:
    """Fraction of as-of out-transfers that went to a prior sender.

    The 2-cycle signal at account level: an account that mostly pays the people
    who pay it is behaving like a layering node, not a merchant.
    """
    idx = _as_of_index(state.out_times, query)
    recip = state.out_cumrecip[idx].astype(np.float64)
    return recip / np.maximum(idx.astype(np.float64), 1.0)


def window_counts(
    state: AccountState, query: np.ndarray, direction: str, window_h: float
) -> tuple[np.ndarray, np.ndarray]:
    """Event count and volume within ``window_h`` hours before each query time.

    Computed as the difference of two as-of prefix positions, so it stays exact
    and costs two searchsorted calls regardless of window size.
    """
    if direction == "out":
        times, cumvol = state.out_times, state.out_cumvol
    else:
        times, cumvol = state.in_times, state.in_cumvol

    hi = _as_of_index(times, query)
    lo = _as_of_index(times, query - window_h)
    return (hi - lo).astype(np.float64), cumvol[hi] - cumvol[lo]


def since_previous_event(state: AccountState, query: np.ndarray) -> np.ndarray:
    """Hours since the account's previous event in either direction.

    Inter-arrival time is how burst behaviour shows up per transaction, and it
    is exactly what the adversary's delay and dormancy genes manipulate.
    Accounts with no prior event get a large sentinel rather than NaN, so
    LightGBM reads "no recent activity" instead of "missing".
    """
    times = state.all_times
    if times.size == 0:
        return np.full(len(query), 1e6, dtype=np.float64)
    idx = _as_of_index(times, query)
    prev = np.where(idx > 0, times[np.maximum(idx - 1, 0)], -1e6)
    gap = query - prev
    return np.where(idx > 0, gap, 1e6)


def pair_repetition(state: AccountState, query: np.ndarray, counterparty: np.ndarray) -> np.ndarray:
    """How many earlier transfers this account made to the same counterparty.

    Repeatedly hitting one counterparty is a mule-reuse signal, which is
    directly what the adversary's ``mule_reuse`` gene trades off.
    """
    n = len(query)
    if state.out_times.size == 0:
        return np.zeros(n, dtype=np.float64)

    out = np.zeros(n, dtype=np.float64)
    idx = _as_of_index(state.out_times, query)
    for i in range(n):
        k = idx[i]
        if k == 0:
            continue
        # The prior array already holds the running per-counterparty count, so
        # find this counterparty's most recent earlier event.
        cps = state.out_cps[:k]
        hits = np.nonzero(cps == counterparty[i])[0]
        if hits.size:
            out[i] = state.out_pair_prior[hits[-1]] + 1.0
    return out
