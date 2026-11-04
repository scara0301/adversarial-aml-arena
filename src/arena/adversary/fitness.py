"""The fitness function, and the guards that keep it honest.

Objective 3 asks for "a fitness function balancing detection evasion against
completion of the adversary's underlying laundering objective". The formula:

    scores     = detector.predict( featurize(sub_ledger, graph.overlay(sub_ledger)) )
    flagged    = scores >= tau
    evasion    = 1 - sum(amount_i * flagged_i) / sum(amount_i)
    completion = delivered_to_destination_by_deadline / amount_goal
    cost       = fees / amount_goal + lambda * elapsed / deadline

    fitness    = completion^alpha * evasion^beta - gamma * cost
                 subject to  completion < floor  =>  fitness = 0

Writing down a formula is the easy part. The hard part is that a genetic
algorithm is an adversarial search over *your own scoring function*, and it will
find every way to score well without doing the task. Four properties of the
formula above exist solely to close those loopholes, and each was chosen against
a specific failure:

**Multiplicative, not additive.** Under ``completion + evasion`` the best
strategy is to move nothing: zero transactions are never flagged, so evasion is
a perfect 1.0. Multiplying means completion 0 gives fitness 0 regardless of how
invisible the strategy is.

**A hard completion floor on top of that.** Multiplication alone still rewards
moving 60% of the money very quietly over 100% slightly less quietly. Since the
adversary's *goal* is to launder the money, a strategy that abandons 40% of it
has not solved the problem. Below the floor, fitness is exactly zero.

**Amount-weighted evasion.** Under an unweighted flag rate, the strongest
strategy is one enormous transfer: a single transaction is one flag opportunity,
so a 50% detection rate on one transaction beats 50% on twenty. Weighting by
amount means flagging the single transfer costs the adversary 100% of its
evasion score. Money noticed is money seized -- the weighting reflects what
actually matters.

**Averaging over several randomised episodes.** A single episode lets the GA
overfit one corner of the background graph: find the account with 3,000
counterparties where anything hides, and route everything through it. That is
not an evasion strategy, it is a property of one dataset. Scoring each genome
against ``K`` independent episodes -- different source, destination, mule pool
and time window -- means only strategies that generalise survive. This is the
most expensive guard and the most important one.

Together these are what stop the "arms race" from being a race between the
detector and a bug in this file.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from arena.adversary.constraints import repair_genome, validate_sub_ledger
from arena.adversary.decoder import (
    DEFAULT_FEE_RATE,
    Episode,
    decode,
    delivered_amount,
    elapsed_hours,
    total_fees,
)
from arena.detector.model import Detector
from arena.features.account_graph import AccountGraph


@dataclass(frozen=True, slots=True)
class FitnessConfig:
    """Fitness weights. Defaults mirror ``config/default.yaml``."""

    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 0.1
    lambda_time: float = 0.3
    completion_floor: float = 0.95
    fee_rate: float = DEFAULT_FEE_RATE


@dataclass(slots=True)
class Outcome:
    """Everything one episode evaluation produced.

    Kept whole rather than collapsed to a scalar because the components are what
    the Phase 2 report and the dashboard actually show: a fitness of 0.6 is
    uninformative, while "completion 1.00, evasion 0.62" says what the strategy
    is doing.
    """

    fitness: float
    completion: float
    evasion: float
    cost: float
    n_transactions: int
    n_flagged: int
    flagged_value_share: float
    sub_ledger: pd.DataFrame


def evaluate_episode(
    genome: np.ndarray,
    episode: Episode,
    detector: Detector,
    graph: AccountGraph,
    tau: float,
    rng: np.random.Generator,
    cfg: FitnessConfig = FitnessConfig(),
    validate: bool = False,
) -> Outcome:
    """Score one genome against one episode.

    The whole detector/adversary interface is the two lines that decode a
    sub-ledger and then score it through ``detector.score_ledger`` on an overlay
    graph. The adversary produces transactions; ``featurize`` -- the same one
    that built the training data -- turns them into features. That is invariant
    1, and it is why an evasion score here means something.
    """
    sub = decode(repair_genome(genome), episode, rng, cfg.fee_rate)

    if sub.empty:
        return Outcome(0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, sub)

    if validate:
        validate_sub_ledger(sub, episode)

    # Invariant 1: transactions in, features out, same code path as training.
    scores = detector.score_ledger(sub, graph.overlay(sub))
    flagged = scores >= tau

    amounts = sub["amount_paid"].to_numpy(dtype=np.float64)
    total_value = float(amounts.sum())
    flagged_share = float((amounts * flagged).sum() / total_value) if total_value > 0 else 1.0
    evasion = 1.0 - flagged_share

    completion = delivered_amount(sub, episode, cfg.fee_rate) / max(episode.amount_goal, 1e-9)
    cost = total_fees(sub) / max(episode.amount_goal, 1e-9) + cfg.lambda_time * (
        elapsed_hours(sub, episode) / max(episode.deadline_hours, 1e-9)
    )

    # Guard: the goal is to launder the money, not most of it.
    if completion < cfg.completion_floor:
        fitness = 0.0
    else:
        fitness = (completion**cfg.alpha) * (evasion**cfg.beta) - cfg.gamma * cost
        fitness = max(fitness, 0.0)

    return Outcome(
        fitness=float(fitness),
        completion=float(completion),
        evasion=float(evasion),
        cost=float(cost),
        n_transactions=int(len(sub)),
        n_flagged=int(flagged.sum()),
        flagged_value_share=float(flagged_share),
        sub_ledger=sub,
    )


def evaluate(
    genome: np.ndarray,
    episodes: list[Episode],
    detector: Detector,
    graph: AccountGraph,
    tau: float,
    rng: np.random.Generator,
    cfg: FitnessConfig = FitnessConfig(),
) -> tuple[float, dict]:
    """Mean fitness over several episodes, plus averaged diagnostics.

    The averaging is the anti-overfitting guard described in the module
    docstring. It multiplies evaluation cost by ``len(episodes)``, which is why
    the copy-on-write overlay in :mod:`arena.features.account_graph` matters so
    much -- without it this would be unaffordable.
    """
    outcomes = [
        evaluate_episode(genome, ep, detector, graph, tau, rng, cfg) for ep in episodes
    ]

    def mean(attr: str) -> float:
        return float(np.mean([getattr(o, attr) for o in outcomes]))

    n_txn = float(np.sum([o.n_transactions for o in outcomes]))
    n_flag = float(np.sum([o.n_flagged for o in outcomes]))

    return mean("fitness"), {
        "fitness": mean("fitness"),
        "completion": mean("completion"),
        "evasion": mean("evasion"),
        "cost": mean("cost"),
        "n_transactions": mean("n_transactions"),
        "flagged_value_share": mean("flagged_value_share"),
        # Unweighted flag rate on the adversary's own transactions. This is the
        # number directly comparable to the detector's recall on the holdout,
        # and it is the Phase 2 headline: naive strategy vs evolved strategy.
        "detector_recall": (n_flag / n_txn) if n_txn > 0 else 0.0,
        # Worst case across episodes: a strategy that is excellent on four
        # episodes and useless on the fifth is fragile, and this surfaces it.
        "min_fitness": float(np.min([o.fitness for o in outcomes])),
    }


class EpisodeSampler:
    """Draws independent laundering tasks, with the ledger scan done once.

    Source, destination and mules are all real accounts from the background, and
    the start time comes from the ledger's own span. Using real accounts is a
    plausibility requirement rather than a convenience: an adversary able to
    invent fresh accounts would evade every account-history feature for free,
    which would be a hole in the simulation instead of a discovered strategy.

    **Why this is a class.** The obvious version is a function that derives the
    account list, the account-to-bank map and the time range from the ledger on
    each call. That made every fitness evaluation O(rows) -- a full scan plus a
    ``drop_duplicates`` over 2M rows -- and measured at 0.19s per candidate,
    against ~8ms for the featurization it was supposed to be supporting. Over a
    20,000-evaluation run that is the difference between three minutes and over
    an hour, and it would have been invisible as anything but "the GA is slow".
    Hoisting the scan into construction leaves ``sample`` doing nothing but
    drawing random indices.

    Each ``sample`` call still returns a *different* set of episodes, which is
    what makes the episode-averaging guard in this module work.
    """

    __slots__ = ("accounts", "bank_of", "lo", "hi", "epoch")

    def __init__(self, ledger: pd.DataFrame, epoch: pd.Timestamp) -> None:
        from_acc = ledger["from_account"].astype("string")
        to_acc = ledger["to_account"].astype("string")

        accounts = pd.concat([from_acc, to_acc]).dropna().unique()
        self.accounts = np.asarray([str(a) for a in accounts], dtype=object)

        self.bank_of: dict[str, int] = {}
        for acc_col, bank_col in ((from_acc, "from_bank"), (to_acc, "to_bank")):
            pairs = pd.DataFrame({"a": acc_col, "b": ledger[bank_col]}).drop_duplicates("a")
            self.bank_of.update(
                dict(zip(pairs["a"].astype("string").tolist(), pairs["b"].astype(int).tolist()))
            )

        hours = (pd.to_datetime(ledger["timestamp"]) - epoch).dt.total_seconds().to_numpy() / 3600.0
        # Episodes start in the first 70% of the span so that a strategy has
        # room to run to its deadline inside the observed ledger.
        self.lo = float(np.percentile(hours, 5))
        self.hi = float(np.percentile(hours, 70))
        self.epoch = epoch

    def sample(
        self,
        n: int,
        rng: np.random.Generator,
        amount_goal: float | tuple[float, float],
        deadline_hours: float,
        mule_pool_size: int = 12,
        start_range: tuple[float, float] | None = None,
    ) -> list[Episode]:
        """Draw ``n`` independent episodes.

        ``start_range`` overrides the default start-time window (hours since the
        epoch). Phase 3 uses it to draw training, validation and evaluation
        episodes from *different* periods of the ledger, so that the adversary
        data the detector learns from and the adversary data it is judged on
        never overlap in time -- the same reason every split in this project is
        chronological.

        ``amount_goal`` is either one fixed amount (Phase 2) or a ``(low, high)``
        range, from which each episode draws its own goal log-uniformly, to the
        cent. Phase 3 needs the range, for two reasons the first closed-loop run
        exposed:

        * **A fixed goal is a round number.** With one tranche the decoder sends
          exactly the goal, so a constant $250,000.00 handed the GA a free
          evasion: the generator's laundering is never round while 21.6% of benign
          traffic is, so the detector had learned "round means benign". The GA
          found it in round 1 (detector recall 0.000).
        * **A fixed goal is a constant the retrained detector can memorise.** If
          every adversary operation moves the same total, the volume features
          encode it, and "recall on held-out episodes" partly measures having
          memorised a number rather than having learned a strategy.
        """
        if isinstance(amount_goal, tuple):
            goal_lo, goal_hi = amount_goal
            if not 0 < goal_lo < goal_hi:
                raise ValueError(f"amount_goal range must satisfy 0 < low < high, got {amount_goal}")
        else:
            goal_lo = goal_hi = None
        lo, hi = start_range if start_range is not None else (self.lo, self.hi)
        if hi <= lo:
            raise ValueError(
                f"empty episode window [{lo:.1f}h, {hi:.1f}h]: the period is shorter than "
                "the episode deadline, so no episode can start in it and still finish"
            )
        episodes: list[Episode] = []
        for _ in range(n):
            picked = rng.choice(len(self.accounts), size=mule_pool_size + 2, replace=False)
            chosen = [str(self.accounts[i]) for i in picked]
            goal = (
                float(amount_goal) if goal_lo is None
                else round(float(np.exp(rng.uniform(np.log(goal_lo), np.log(goal_hi)))), 2)
            )
            episodes.append(
                Episode(
                    src_account=chosen[0],
                    dst_account=chosen[1],
                    amount_goal=goal,
                    deadline_hours=deadline_hours,
                    start_hour=float(rng.uniform(lo, hi)),
                    mule_pool=chosen[2:],
                    bank_of=self.bank_of,
                    epoch=self.epoch,
                )
            )
        return episodes


def sample_episodes(
    ledger: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
    amount_goal: float,
    deadline_hours: float,
    epoch: pd.Timestamp,
    mule_pool_size: int = 12,
) -> list[Episode]:
    """Convenience wrapper around :class:`EpisodeSampler` for one-off draws.

    Rebuilds the sampler each call, so never use it inside a loop -- see the
    performance note on :class:`EpisodeSampler`.
    """
    return EpisodeSampler(ledger, epoch).sample(
        n, rng, amount_goal, deadline_hours, mule_pool_size
    )
