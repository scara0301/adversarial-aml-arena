"""Account graph: frozen snapshot plus copy-on-write overlay.

This module exists to solve one specific problem, and it is the reason the whole
project is feasible.

A GA run is population 100 x 40 generations x 5 episodes ~= 20,000 candidate
evaluations. Each candidate is a sub-ledger of a few dozen transactions that must
be scored by the detector. But the detector's features are *relational*: account
degree, rolling volume, reciprocity, cycle membership. Those depend on the
surrounding ledger. Re-featurizing a multi-million-row background ledger 20,000
times is impossible.

So:

* :meth:`AccountGraph.build` does one pass over the background ledger, once per
  arms-race round, and stores per-account time-sorted event arrays with prefix
  sums.
* :meth:`AccountGraph.overlay` returns a copy-on-write view. Only the accounts
  the sub-ledger touches -- tens of them, bounded by hops x mules -- get rebuilt
  state. Every other account reads through to the frozen snapshot.

Because both paths hand the *same* state objects to the *same* feature functions,
batch and overlay featurization agree by construction, not by coincidence. That
is invariant 1 from CLAUDE.md, and ``tests/test_features.py`` checks it holds.

## Everything here is "as of" the transaction's own timestamp

Account state is always evaluated using only events at or before the transaction
being featurized. This is not a detail. A feature like "this account's total
out-volume" computed over the whole ledger leaks the future into the past: the
detector would learn from transactions that had not happened yet, and every
metric would be inflated. As-of semantics also make the overlay exact, since an
injected sub-ledger cannot retroactively change an account's earlier state.

The one deliberate exception is the cycle index -- see :class:`CycleIndex`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: Cap on neighbours considered per account during cycle search. Hub accounts can
#: have thousands of counterparties, which would make the search quadratic. The
#: cap keeps it bounded, and taking the *most recent* neighbours makes it
#: deterministic -- essential, because batch and overlay must agree exactly.
NEIGHBOUR_CAP = 32

#: Cap on the precomputed 2-hop out-set per account, used for 4-cycle detection.
TWO_HOP_CAP = 256

_EMPTY_F = np.zeros(0, dtype=np.float64)
_EMPTY_I = np.zeros(0, dtype=np.int64)


@dataclass(slots=True)
class AccountState:
    """Time-sorted event history for one account, with prefix sums.

    Split by direction so that an as-of query is a single ``searchsorted``.
    Arrays are parallel: ``out_times[i]`` pairs with ``out_amounts[i]`` and
    ``out_cps[i]``.

    The three ``*_flag`` / ``*_prior`` arrays are precomputed during the build
    walk because they cannot be derived from a searchsorted alone:

    ``out_first_cp``
        1 if this out-event was the account's first ever transfer to that
        counterparty. Prefix-summing gives the as-of unique-counterparty count.
    ``out_recip``
        1 if, at that moment, the counterparty had already sent *to* this
        account. Prefix-summing gives as-of reciprocity, the 2-cycle signal.
    ``out_pair_prior``
        How many earlier transfers this account had already made to that same
        counterparty. Repetition on a directed pair is a layering signal.
    """

    out_times: np.ndarray = field(default_factory=lambda: _EMPTY_F)
    out_amounts: np.ndarray = field(default_factory=lambda: _EMPTY_F)
    out_cps: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=object))
    out_cumvol: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    out_cumfirst: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.int64))
    out_cumrecip: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.int64))
    out_pair_prior: np.ndarray = field(default_factory=lambda: _EMPTY_I)

    in_times: np.ndarray = field(default_factory=lambda: _EMPTY_F)
    in_amounts: np.ndarray = field(default_factory=lambda: _EMPTY_F)
    in_cps: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=object))
    in_cumvol: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    in_cumfirst: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.int64))

    first_seen: float = np.inf

    # Direction-merged event times, for inter-arrival.
    all_times: np.ndarray = field(default_factory=lambda: _EMPTY_F)


def _build_state(
    out_t: list[float],
    out_a: list[float],
    out_c: list[str],
    in_t: list[float],
    in_a: list[float],
    in_c: list[str],
) -> AccountState:
    """Assemble one account's state, including the single ordered walk.

    The walk is what makes reciprocity, unique-counterparty and pair-repetition
    counts exact under as-of semantics. It is O(events) per account and runs once
    per graph build.
    """
    o_order = np.argsort(np.asarray(out_t, dtype=np.float64), kind="stable")
    i_order = np.argsort(np.asarray(in_t, dtype=np.float64), kind="stable")

    out_times = np.asarray(out_t, dtype=np.float64)[o_order]
    out_amounts = np.asarray(out_a, dtype=np.float64)[o_order]
    out_cps = np.asarray(out_c, dtype=object)[o_order]
    in_times = np.asarray(in_t, dtype=np.float64)[i_order]
    in_amounts = np.asarray(in_a, dtype=np.float64)[i_order]
    in_cps = np.asarray(in_c, dtype=object)[i_order]

    n_out = len(out_times)
    out_first = np.zeros(n_out, dtype=np.int64)
    out_recip = np.zeros(n_out, dtype=np.int64)
    out_pair_prior = np.zeros(n_out, dtype=np.int64)

    # Merge out- and in-events in time order, maintaining the sets that the
    # as-of flags depend on. Ties resolve to incoming-first, which is the
    # conservative choice: an inbound transfer arriving at the same instant is
    # treated as already known.
    seen_out: set = set()
    seen_in: set = set()
    pair_counts: Counter = Counter()
    oi = ii = 0
    while oi < n_out:
        while ii < len(in_times) and in_times[ii] <= out_times[oi]:
            seen_in.add(in_cps[ii])
            ii += 1
        cp = out_cps[oi]
        out_first[oi] = 0 if cp in seen_out else 1
        out_recip[oi] = 1 if cp in seen_in else 0
        out_pair_prior[oi] = pair_counts[cp]
        seen_out.add(cp)
        pair_counts[cp] += 1
        oi += 1

    in_first = np.zeros(len(in_times), dtype=np.int64)
    seen: set = set()
    for k, cp in enumerate(in_cps):
        in_first[k] = 0 if cp in seen else 1
        seen.add(cp)

    first_seen = float(min(out_times[0] if n_out else np.inf, in_times[0] if len(in_times) else np.inf))

    return AccountState(
        out_times=out_times,
        out_amounts=out_amounts,
        out_cps=out_cps,
        out_cumvol=np.concatenate([[0.0], np.cumsum(out_amounts)]),
        out_cumfirst=np.concatenate([[0], np.cumsum(out_first)]),
        out_cumrecip=np.concatenate([[0], np.cumsum(out_recip)]),
        out_pair_prior=out_pair_prior,
        in_times=in_times,
        in_amounts=in_amounts,
        in_cps=in_cps,
        in_cumvol=np.concatenate([[0.0], np.cumsum(in_amounts)]),
        in_cumfirst=np.concatenate([[0], np.cumsum(in_first)]),
        first_seen=first_seen,
        all_times=np.sort(np.concatenate([out_times, in_times])),
    )


class CycleIndex:
    """Directed adjacency supporting bounded cycle queries.

    Answers "is this transfer closing a short cycle?" for cycle lengths 2, 3
    and 4 -- the cycle typology's signature, and one of the five typologies
    objective 1 names.

    **Deliberate deviation from as-of semantics.** This index uses cumulative
    topology rather than as-of topology. Maintaining time-sliced reachability
    for every account would cost far more than the feature is worth, and the
    leak is mild: it exposes graph structure, never labels. Both the batch and
    overlay paths use the identical cumulative index, so the equivalence
    invariant still holds. This is recorded as a known limitation in
    ``docs/phase1.md`` rather than hidden.
    """

    __slots__ = ("out_adj", "in_adj", "_two_hop")

    def __init__(self, out_adj: dict[str, list[str]], in_adj: dict[str, list[str]]) -> None:
        # Keep only the most recent NEIGHBOUR_CAP neighbours: bounded and
        # deterministic, so batch and overlay cannot diverge on hub accounts.
        self.out_adj = {k: set(v[-NEIGHBOUR_CAP:]) for k, v in out_adj.items()}
        self.in_adj = {k: set(v[-NEIGHBOUR_CAP:]) for k, v in in_adj.items()}
        self._two_hop: dict[str, set[str]] = {}

    @classmethod
    def build(cls, from_acc: np.ndarray, to_acc: np.ndarray) -> CycleIndex:
        out_adj: dict[str, list[str]] = {}
        in_adj: dict[str, list[str]] = {}
        for f, t in zip(from_acc, to_acc):
            out_adj.setdefault(f, []).append(t)
            in_adj.setdefault(t, []).append(f)
        return cls(out_adj, in_adj)

    def extended(self, from_acc: np.ndarray, to_acc: np.ndarray) -> CycleIndex:
        """A copy with extra edges, for the overlay path.

        Only the touched accounts' neighbour sets are copied; the rest are
        shared with this index. Cheap, and safe because neighbour sets are never
        mutated in place after construction.
        """
        new = CycleIndex.__new__(CycleIndex)
        new.out_adj = dict(self.out_adj)
        new.in_adj = dict(self.in_adj)
        new._two_hop = {}
        for f, t in zip(from_acc, to_acc):
            new.out_adj[f] = set(new.out_adj.get(f, ())) | {t}
            new.in_adj[t] = set(new.in_adj.get(t, ())) | {f}
        return new

    def two_hop_out(self, node: str) -> set[str]:
        """Accounts reachable in exactly two out-hops, capped and memoised."""
        cached = self._two_hop.get(node)
        if cached is not None:
            return cached
        acc: set[str] = set()
        for mid in self.out_adj.get(node, ()):
            acc |= self.out_adj.get(mid, set())
            if len(acc) >= TWO_HOP_CAP:
                break
        result = set(list(acc)[:TWO_HOP_CAP])
        self._two_hop[node] = result
        return result

    def cycle_flags(self, from_acc: np.ndarray, to_acc: np.ndarray) -> dict[str, np.ndarray]:
        """For each transfer f->t, does it close a cycle of length 2, 3 or 4?

        * length 2: t already sends to f.
        * length 3: some X with t->X and X->f.
        * length 4: some X,Y with t->X->Y and Y->f.

        Each is a set intersection against capped neighbour sets, so the cost
        per row is bounded regardless of how hub-like the accounts are.
        """
        n = len(from_acc)
        c2 = np.zeros(n, dtype=np.int8)
        c3 = np.zeros(n, dtype=np.int8)
        c4 = np.zeros(n, dtype=np.int8)

        for i in range(n):
            f = from_acc[i]
            t = to_acc[i]
            t_out = self.out_adj.get(t)
            if not t_out:
                continue
            if f in t_out:
                c2[i] = 1
            f_in = self.in_adj.get(f)
            if not f_in:
                continue
            if not t_out.isdisjoint(f_in):
                c3[i] = 1
            if not self.two_hop_out(t).isdisjoint(f_in):
                c4[i] = 1
        return {"cycle2": c2, "cycle3": c3, "cycle4": c4}


class AccountGraph:
    """Per-account as-of state plus a cycle index, with cheap overlays."""

    __slots__ = ("states", "cycles", "epoch", "_default")

    def __init__(
        self,
        states: dict[str, AccountState],
        cycles: CycleIndex,
        epoch: pd.Timestamp,
    ) -> None:
        self.states = states
        self.cycles = cycles
        self.epoch = epoch
        # Shared immutable fallback for accounts with no history. Returning this
        # rather than raising means the adversary may route through a previously
        # unseen account without crashing featurization.
        self._default = AccountState()

    # ---------------------------------------------------------------- building

    @classmethod
    def build(cls, ledger: pd.DataFrame, epoch: pd.Timestamp | None = None) -> AccountGraph:
        """One O(N) pass over ``ledger``. Run once per arms-race round."""
        if epoch is None:
            epoch = pd.Timestamp(ledger["timestamp"].min()) if len(ledger) else pd.Timestamp("2026-01-01")

        hours = to_hours(ledger["timestamp"], epoch)
        from_acc = ledger["from_account"].astype("string").to_numpy(dtype=object)
        to_acc = ledger["to_account"].astype("string").to_numpy(dtype=object)
        amounts = ledger["amount_paid"].to_numpy(dtype=np.float64)

        out_t: dict[str, list[float]] = {}
        out_a: dict[str, list[float]] = {}
        out_c: dict[str, list[str]] = {}
        in_t: dict[str, list[float]] = {}
        in_a: dict[str, list[float]] = {}
        in_c: dict[str, list[str]] = {}

        for i in range(len(hours)):
            f, t, h, amt = from_acc[i], to_acc[i], hours[i], amounts[i]
            out_t.setdefault(f, []).append(h)
            out_a.setdefault(f, []).append(amt)
            out_c.setdefault(f, []).append(t)
            in_t.setdefault(t, []).append(h)
            in_a.setdefault(t, []).append(amt)
            in_c.setdefault(t, []).append(f)

        states: dict[str, AccountState] = {}
        for acct in set(out_t) | set(in_t):
            states[acct] = _build_state(
                out_t.get(acct, []),
                out_a.get(acct, []),
                out_c.get(acct, []),
                in_t.get(acct, []),
                in_a.get(acct, []),
                in_c.get(acct, []),
            )

        return cls(states, CycleIndex.build(from_acc, to_acc), epoch)

    def overlay(self, sub_ledger: pd.DataFrame) -> AccountGraph:
        """A copy-on-write view with ``sub_ledger``'s events merged in.

        Rebuilds state only for accounts the sub-ledger touches -- typically a
        few dozen. Untouched accounts are shared with this graph by reference,
        so an overlay costs microseconds even against a multi-million-row
        background.
        """
        if sub_ledger.empty:
            return self

        hours = to_hours(sub_ledger["timestamp"], self.epoch)
        from_acc = sub_ledger["from_account"].astype("string").to_numpy(dtype=object)
        to_acc = sub_ledger["to_account"].astype("string").to_numpy(dtype=object)
        amounts = sub_ledger["amount_paid"].to_numpy(dtype=np.float64)

        touched = set(from_acc) | set(to_acc)
        add_out: dict[str, tuple[list, list, list]] = {a: ([], [], []) for a in touched}
        add_in: dict[str, tuple[list, list, list]] = {a: ([], [], []) for a in touched}

        for i in range(len(hours)):
            f, t, h, amt = from_acc[i], to_acc[i], hours[i], amounts[i]
            add_out[f][0].append(h)
            add_out[f][1].append(amt)
            add_out[f][2].append(t)
            add_in[t][0].append(h)
            add_in[t][1].append(amt)
            add_in[t][2].append(f)

        states = dict(self.states)  # shallow: untouched accounts shared
        for acct in touched:
            base = self.states.get(acct, self._default)
            o_t, o_a, o_c = add_out[acct]
            i_t, i_a, i_c = add_in[acct]
            states[acct] = _build_state(
                list(base.out_times) + o_t,
                list(base.out_amounts) + o_a,
                list(base.out_cps) + o_c,
                list(base.in_times) + i_t,
                list(base.in_amounts) + i_a,
                list(base.in_cps) + i_c,
            )

        return AccountGraph(states, self.cycles.extended(from_acc, to_acc), self.epoch)

    # ----------------------------------------------------------------- lookups

    def state(self, account: str) -> AccountState:
        return self.states.get(account, self._default)

    @property
    def n_accounts(self) -> int:
        return len(self.states)


def to_hours(timestamps: pd.Series, epoch: pd.Timestamp) -> np.ndarray:
    """Timestamps as float hours since ``epoch``.

    A single shared time representation. Hours (not seconds) because every
    window in the config is expressed in hours, and float64 keeps sub-second
    precision over a multi-year span.
    """
    delta = pd.to_datetime(timestamps).to_numpy(dtype="datetime64[ns]") - np.datetime64(epoch)
    return delta.astype("timedelta64[s]").astype(np.float64) / 3600.0
