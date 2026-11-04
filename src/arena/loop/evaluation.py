"""Scoring a strategy against a detector, comparably across rounds.

The arms-race curve is a comparison between many (detector, strategy) pairs, so
the one thing this module must guarantee is that any two cells of it differ
*only* because the detector or the strategy differs -- never because of the
dice. Two choices deliver that:

* **A fixed episode set.** Every pair is scored on the same laundering tasks
  (same accounts, mules and start times), drawn once from a period of the
  ledger no detector was trained on.
* **A fixed decode seed.** ``decode`` is stochastic (Dirichlet tranche weights,
  gamma delays). Re-seeding it identically for every pair means a given genome
  decodes to the *same transactions* whichever detector scores it.

Without the second, a detector could look better against a strategy simply
because it drew an easier sample of that strategy's randomness.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from arena.adversary.decoder import Episode
from arena.adversary.fitness import FitnessConfig, evaluate_episode
from arena.detector.model import Detector
from arena.features import AccountGraph

#: A single episode counts as an adversary success when the money arrives and at
#: least this share of its value escaped the alert queue. Half is a deliberately
#: plain bar: more than half the laundered value got through unflagged.
SUCCESS_EVASION = 0.5


@dataclass(slots=True)
class StrategyScore:
    """One strategy scored against one detector over a set of episodes."""

    detector_recall: float   # flagged transactions / all transactions, pooled
    evasion: float           # amount-weighted share of value left unflagged, mean over episodes
    completion: float        # share of the goal delivered by the deadline, mean over episodes
    fitness: float           # the GA's own objective, mean over episodes
    success_rate: float      # fraction of episodes with completion >= floor and evasion >= SUCCESS_EVASION
    n_transactions: float    # mean transactions per episode

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


@dataclass(slots=True)
class EpisodeOutcomes:
    """Per-episode results for one strategy against one detector.

    Kept per episode (not just averaged) because the right error bar for a
    strategy that sends a handful of transactions comes from resampling
    *episodes*. A one-transaction strategy is close to a coin flip each episode:
    flagged or not, with no middle. Averaged over 30 episodes that leaves a
    standard error near 0.09, which is larger than most round-to-round changes
    one would want to read into.
    """

    n_flagged: np.ndarray
    n_transactions: np.ndarray
    evasion: np.ndarray
    completion: np.ndarray
    fitness: np.ndarray
    success: np.ndarray


def episode_outcomes(
    genome: np.ndarray,
    episodes: list[Episode],
    detector: Detector,
    graph: AccountGraph,
    tau: float,
    fitness_cfg: FitnessConfig,
    seed: int = 0,
) -> EpisodeOutcomes:
    """Score ``genome`` on every episode against ``detector`` at threshold ``tau``."""
    rng = np.random.default_rng(seed)
    outs = [evaluate_episode(genome, ep, detector, graph, tau, rng, fitness_cfg) for ep in episodes]
    return EpisodeOutcomes(
        n_flagged=np.array([o.n_flagged for o in outs], dtype=float),
        n_transactions=np.array([o.n_transactions for o in outs], dtype=float),
        evasion=np.array([o.evasion for o in outs]),
        completion=np.array([o.completion for o in outs]),
        fitness=np.array([o.fitness for o in outs]),
        success=np.array([
            o.completion >= fitness_cfg.completion_floor and o.evasion >= SUCCESS_EVASION for o in outs
        ]),
    )


def _recall(o: EpisodeOutcomes, idx: np.ndarray | slice = slice(None)) -> float:
    n = float(o.n_transactions[idx].sum())
    return float(o.n_flagged[idx].sum() / n) if n else 0.0


def summarize(o: EpisodeOutcomes) -> StrategyScore:
    return StrategyScore(
        detector_recall=_recall(o),
        evasion=float(o.evasion.mean()),
        completion=float(o.completion.mean()),
        fitness=float(o.fitness.mean()),
        success_rate=float(o.success.mean()),
        n_transactions=float(o.n_transactions.mean()),
    )


def bootstrap_ci(
    o: EpisodeOutcomes, rng: np.random.Generator, n_boot: int = 1000, level: float = 0.95
) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap over *episodes*, for detector recall and evasion."""
    n = len(o.evasion)
    idx = rng.integers(0, n, size=(n_boot, n))
    recall = np.array([_recall(o, i) for i in idx])
    evasion = o.evasion[idx].mean(axis=1)
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    return {
        "detector_recall": (float(np.percentile(recall, lo)), float(np.percentile(recall, hi))),
        "evasion": (float(np.percentile(evasion, lo)), float(np.percentile(evasion, hi))),
    }


def score_strategy(
    genome: np.ndarray,
    episodes: list[Episode],
    detector: Detector,
    graph: AccountGraph,
    tau: float,
    fitness_cfg: FitnessConfig,
    seed: int = 0,
) -> StrategyScore:
    """Mean score of ``genome`` over ``episodes`` against ``detector`` at ``tau``."""
    return summarize(episode_outcomes(genome, episodes, detector, graph, tau, fitness_cfg, seed))


def score_pool(
    genomes: list[np.ndarray],
    episodes: list[Episode],
    detector: Detector,
    graph: AccountGraph,
    tau: float,
    fitness_cfg: FitnessConfig,
    seed: int = 0,
) -> StrategyScore:
    """Mean score over a set of strategies -- the adversary *population*, not
    just its champion. The champion answers "can the adversary still win?"; the
    pool answers "how much of what it knows still works?"."""
    scores = [score_strategy(g, episodes, detector, graph, tau, fitness_cfg, seed) for g in genomes]
    fields = scores[0].as_dict().keys()
    return StrategyScore(**{k: float(np.mean([s.as_dict()[k] for s in scores])) for k in fields})
