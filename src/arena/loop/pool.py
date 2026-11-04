"""The strategy pool: everything the adversary has shown the detector so far.

This module is the feedback half of the closed loop, and almost every guard
against the failure modes that retraining on adversary output invites lives
here. The failure modes, and the guard each one gets:

**Detector collapse / catastrophic forgetting.** Flooding the training set with
adversary positives drags the decision boundary toward the adversary and away
from real laundering. So adversary rows are *capped* at a fixed fraction of the
original positive count (default 20%): the detector learns from the adversary
without being able to forget the data it was built on. Retraining always starts
from the **original** training set plus the pool, never from the previous
round's already-augmented set -- augmenting an augmented set compounds drift
round over round.

**Forgetting old adversaries.** If the pool kept only the latest round's
strategies, a detector could trade away what it learned about round 1 in order
to fit round 4. So the cap is split into equal *row* slices, one per round the
pool holds, and old evasion strategies stay in the training mix for good. (Any
slice a round cannot fill is handed on to the rounds that can.)

**Overfitting to GA artefacts.** One decoded strategy on one episode is one
laundering operation; train on a dozen copies of it and the detector learns
that operation's account ids and timestamps, not the strategy. So every
strategy is decoded on several *different* episodes (different accounts,
mules and start times) before it is used, and the pool holds the GA's
mutually-distinct top strategies (``distinct_top``) rather than its top-k by
fitness, which would be near-clones.

**Time leakage.** Adversary rows for the training set are decoded only from
episodes that start *and finish* inside the training period, and validation
rows only inside the validation period -- the same chronological discipline as
every other split here. Whole laundering operations are kept or dropped, never
cut: half a fan-out is not a laundering pattern, it is noise with a positive
label.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from arena.adversary.decoder import Episode, decode
from arena.adversary.fitness import EpisodeSampler
from arena.features import AccountGraph, featurize

#: Adversary pattern ids start here, so they can never collide with the
#: background generator's ids (which count up from 0).
ADVERSARY_PATTERN_ID_BASE = 1_000_000


@dataclass(slots=True)
class Materialized:
    """Adversary training rows, ready to append to a detector's training set."""

    X: pd.DataFrame
    rows: pd.DataFrame
    n_instances: int
    rows_per_round: dict[int, int] = field(default_factory=dict)
    dropped_out_of_window: int = 0
    dropped_over_cap: int = 0

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def y(self) -> np.ndarray:
        return np.ones(len(self.X), dtype=int)


class StrategyPool:
    """Cumulative record of the evasive strategies shown to the detector."""

    def __init__(self) -> None:
        self._rounds: dict[int, list[np.ndarray]] = {}

    def add_round(self, round_idx: int, genomes: list[np.ndarray]) -> None:
        self._rounds[int(round_idx)] = [np.asarray(g, dtype=np.float64).copy() for g in genomes]

    @property
    def rounds(self) -> list[int]:
        return sorted(self._rounds)

    def __len__(self) -> int:
        return sum(len(g) for g in self._rounds.values())

    def genomes(self, round_idx: int) -> list[np.ndarray]:
        return list(self._rounds[round_idx])

    def recent_genomes(self, limit: int) -> list[np.ndarray]:
        """Up to ``limit`` genomes, newest round first. Used to warm-start the GA."""
        out: list[np.ndarray] = []
        for r in reversed(self.rounds):
            for g in self._rounds[r]:
                if len(out) >= limit:
                    return out
                out.append(g.copy())
        return out

    def materialize(
        self,
        *,
        sampler: EpisodeSampler,
        graph: AccountGraph,
        window_hours: tuple[float, ...],
        rng: np.random.Generator,
        window: tuple[float, float],
        row_cap: int,
        instances_per_strategy: int,
        amount_goal: float | tuple[float, float],
        deadline_hours: float,
        fee_rate: float,
        id_base: int = ADVERSARY_PATTERN_ID_BASE,
    ) -> Materialized:
        """Decode the pool into labelled laundering instances under a row cap.

        Args:
            window: ``(start_hour, end_hour)`` of the period these rows may come
                from. Episodes start in ``[start, end - deadline]`` and any
                instance that still runs past ``end`` is dropped whole.
            row_cap: Maximum adversary *rows* to return. Shared round-robin
                across the rounds in the pool.
            instances_per_strategy: Episodes each strategy is decoded on.

        Features come from ``featurize(instance, graph.overlay(instance))`` --
        the exact path the GA scored the strategy through (invariant 1) -- one
        instance at a time, so instances never see one another.
        """
        empty = Materialized(X=pd.DataFrame(), rows=pd.DataFrame(), n_instances=0)
        if row_cap <= 0 or not self._rounds:
            return empty

        w_start, w_end = window
        start_range = (w_start, w_end - deadline_hours)

        queues: dict[int, deque] = {}
        out_of_window = 0
        for r in self.rounds:
            instances: list[pd.DataFrame] = []
            for genome in self._rounds[r]:
                episodes = sampler.sample(
                    instances_per_strategy, rng, amount_goal, deadline_hours,
                    start_range=start_range,
                )
                for ep in episodes:
                    sub = decode(genome, ep, rng, fee_rate)
                    if sub.empty:
                        continue
                    if _last_hour(sub, ep) > w_end:
                        out_of_window += 1
                        continue
                    instances.append(sub)
            order = rng.permutation(len(instances))
            queues[r] = deque(instances[i] for i in order)

        chosen: list[pd.DataFrame] = []
        rows_per_round = {r: 0 for r in queues}
        used = 0

        def take(r: int, inst: pd.DataFrame) -> None:
            nonlocal used
            chosen.append(inst.assign(
                pattern_id=pd.array([id_base + len(chosen)] * len(inst), dtype="Int64")
            ))
            used += len(inst)
            rows_per_round[r] += len(inst)

        # Pass 1: every round fills its own equal slice of the cap, in ROWS.
        #
        # The slice is counted in rows, not operations, on purpose. An earlier
        # version took one operation per round per pass, which is equal only if
        # operations are the same size -- and they are not (a decoded strategy
        # runs from 2 to over 100 transactions). One round's large operations
        # filled the cap and a later round got zero rows: a silent failure of the
        # very guard this exists for. First-fit within the slice: an operation
        # that does not fit is skipped, a smaller one may.
        slice_rows = row_cap // len(queues)
        for r, q in queues.items():
            leftover: deque = deque()
            for inst in q:
                if rows_per_round[r] + len(inst) <= slice_rows:
                    take(r, inst)
                else:
                    leftover.append(inst)
            queues[r] = leftover

        # Pass 2: whatever the slices left unused is shared round-robin among the
        # rounds that still have operations, so the cap is not wasted just
        # because one round's strategies were too large to fill its slice.
        over_cap = 0
        while any(queues.values()):
            for r in list(queues):
                if not queues[r]:
                    continue
                inst = queues[r].popleft()
                if used + len(inst) > row_cap:
                    over_cap += 1
                    continue
                take(r, inst)

        if not chosen:
            return Materialized(
                X=pd.DataFrame(), rows=pd.DataFrame(), n_instances=0,
                dropped_out_of_window=out_of_window, dropped_over_cap=over_cap,
            )

        X = pd.concat(
            [featurize(inst, graph.overlay(inst), window_hours) for inst in chosen],
            ignore_index=True,
        )
        rows = pd.concat(chosen, ignore_index=True)
        return Materialized(
            X=X, rows=rows, n_instances=len(chosen), rows_per_round=rows_per_round,
            dropped_out_of_window=out_of_window, dropped_over_cap=over_cap,
        )


def _last_hour(sub: pd.DataFrame, episode: Episode) -> float:
    """Hours since the epoch of the instance's final transaction."""
    return float((pd.to_datetime(sub["timestamp"]).max() - episode.epoch).total_seconds() / 3600.0)
