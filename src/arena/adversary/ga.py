"""The genetic algorithm driver. All DEAP usage is confined to this file.

The confinement is deliberate. ``genome.py``, ``decoder.py``, ``constraints.py``
and ``fitness.py`` are pure NumPy and know nothing about DEAP, so if DEAP ever
becomes a problem -- it is a C-extension package with a slow release cadence, and
the system interpreter here is Python 3.14 where its wheels are uncertain -- the
replacement is a ~150-line loop in this one file and nothing else changes.

The loop is written out rather than delegated to ``deap.algorithms.eaSimple``
because three things are needed that ``eaSimple`` does not provide:

* **Elitism.** Without carrying the best individuals forward unchanged, a good
  strategy can be lost to an unlucky crossover and the best-fitness curve
  sawtooths instead of rising.
* **Fitness sharing.** Left alone, the population converges to clones within
  ~15 generations. That is fine for finding one strategy and useless for
  Phase 3, which needs a *varied* set of evasive strategies to fold back into
  the detector's training data. Sharing keeps niches alive.
* **Per-generation telemetry.** The convergence curve and the component
  breakdown (completion vs evasion) are Phase 2 deliverables, not debug output.

Selection uses *shared* fitness while all reporting uses *raw* fitness, so the
diversity mechanism shapes the search without distorting the numbers.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from deap import base, creator, tools

from arena.adversary import fitness as fit
from arena.adversary.constraints import repair_genome
from arena.adversary.decoder import Episode
from arena.adversary.genome import GENE_INDEX, HIGH, LOW, N_GENES, distance
from arena.detector.model import Detector
from arena.features.account_graph import AccountGraph

#: Niche radius for fitness sharing, in normalised genome distance. Individuals
#: closer than this share credit, which stops one strategy from taking over.
SIGMA_SHARE = 0.15

# DEAP registers classes globally on the creator module. Guarded so repeated
# imports (notebooks, pytest) do not raise.
if not hasattr(creator, "AdversaryFitness"):
    creator.create("AdversaryFitness", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.AdversaryFitness)


@dataclass(slots=True)
class GAConfig:
    """GA hyperparameters. Defaults mirror ``config/default.yaml``."""

    population: int = 100
    generations: int = 40
    tournament_size: int = 3
    crossover_prob: float = 0.7
    mutation_prob: float = 0.2
    elitism: int = 5
    episodes_per_eval: int = 5
    seed: int = 424242


@dataclass(slots=True)
class GenerationRecord:
    """One generation's telemetry."""

    generation: int
    best_fitness: float
    mean_fitness: float
    best_completion: float
    best_evasion: float
    best_n_transactions: float
    diversity: float
    best_genome: list[float] = field(default_factory=list)


def naive_genome() -> np.ndarray:
    """The unsophisticated baseline strategy: just move the money.

    One hop, three chunks, no threshold awareness, no delays, no business-hours
    alignment, no mule reuse strategy. This is roughly what the synthetic
    generator's hand-written typologies do, and it is the reference the evolved
    adversary is measured against. The Phase 2 headline result is the gap
    between the detector's recall on this and on the evolved strategy -- if
    there is no gap, the GA has learned nothing.
    """
    g = np.zeros(N_GENES, dtype=np.float64)
    g[GENE_INDEX["typology_template"]] = 3.0  # stack: plain sequential layering
    g[GENE_INDEX["n_hops"]] = 1.0
    g[GENE_INDEX["n_splits"]] = 3.0
    g[GENE_INDEX["split_skew"]] = 0.5
    g[GENE_INDEX["threshold_hug"]] = 0.0
    g[GENE_INDEX["mean_delay_h"]] = 0.5
    g[GENE_INDEX["delay_jitter"]] = 0.1
    g[GENE_INDEX["business_hours_bias"]] = 0.0
    g[GENE_INDEX["mule_pool_size"]] = 1.0
    g[GENE_INDEX["mule_reuse"]] = 1.0
    g[GENE_INDEX["format_mix"]] = 0.5
    g[GENE_INDEX["dormancy_h"]] = 0.0
    return repair_genome(g)


def _shared_fitness(population: list, raw: np.ndarray) -> np.ndarray:
    """Fitness sharing: divide each individual's credit by its niche count.

    ``sh(d) = 1 - d/sigma`` for ``d < sigma``, else 0. An individual in a crowded
    region of genome space has a large niche count and is discounted, so
    selection spreads across niches instead of piling onto one peak.
    """
    n = len(population)
    genomes = [np.asarray(ind, dtype=np.float64) for ind in population]
    niche = np.ones(n, dtype=np.float64)
    for i in range(n):
        total = 0.0
        for j in range(n):
            d = distance(genomes[i], genomes[j])
            if d < SIGMA_SHARE:
                total += 1.0 - d / SIGMA_SHARE
        niche[i] = max(total, 1.0)
    return raw / niche


def _population_diversity(population: list) -> float:
    """Mean pairwise genome distance. Sampled above 40 individuals to stay O(1)."""
    genomes = [np.asarray(ind, dtype=np.float64) for ind in population]
    if len(genomes) > 40:
        idx = np.linspace(0, len(genomes) - 1, 40).astype(int)
        genomes = [genomes[i] for i in idx]
    if len(genomes) < 2:
        return 0.0
    dists = [
        distance(genomes[i], genomes[j])
        for i in range(len(genomes))
        for j in range(i + 1, len(genomes))
    ]
    return float(np.mean(dists))


def build_toolbox(
    detector: Detector,
    graph: AccountGraph,
    tau: float,
    ledger: pd.DataFrame,
    cfg: GAConfig,
    fitness_cfg: fit.FitnessConfig,
    amount_goal: float,
    deadline_hours: float,
    epoch: pd.Timestamp,
) -> tuple[base.Toolbox, np.random.Generator]:
    """Wire up DEAP's operators.

    Mutation sigma is scaled per gene to 12% of that gene's range, so a mutation
    means the same thing for ``dormancy_h`` (range 168) as for ``split_skew``
    (range 1). A single shared sigma would make some genes effectively frozen
    and others pure noise.
    """
    rng = np.random.default_rng(cfg.seed)
    random.seed(cfg.seed)

    # Scan the ledger once, not once per fitness evaluation.
    sampler = fit.EpisodeSampler(ledger, epoch)

    toolbox = base.Toolbox()
    toolbox.register(
        "individual",
        lambda: creator.Individual(rng.uniform(LOW, HIGH).tolist()),
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)

    sigma = ((HIGH - LOW) * 0.12).tolist()
    toolbox.register("mutate", tools.mutGaussian, mu=[0.0] * N_GENES, sigma=sigma, indpb=0.25)
    toolbox.register("select", tools.selTournament, tournsize=cfg.tournament_size)

    def evaluate(individual) -> tuple[float]:
        # Fresh episodes each call: the anti-overfitting guard from fitness.py.
        episodes = sampler.sample(
            cfg.episodes_per_eval, rng, amount_goal, deadline_hours
        )
        score, diag = fit.evaluate(
            np.asarray(individual, dtype=np.float64), episodes, detector, graph, tau, rng, fitness_cfg
        )
        individual.diagnostics = diag
        return (score,)

    toolbox.register("evaluate", evaluate)
    return toolbox, rng


def evolve(
    detector: Detector,
    graph: AccountGraph,
    tau: float,
    ledger: pd.DataFrame,
    cfg: GAConfig = GAConfig(),
    fitness_cfg: fit.FitnessConfig = fit.FitnessConfig(),
    amount_goal: float = 250_000.0,
    deadline_hours: float = 336.0,
    epoch: pd.Timestamp | None = None,
    progress: bool = True,
) -> tuple[np.ndarray, list[GenerationRecord], list]:
    """Run the GA and return the best genome, the per-generation log, and the
    final population.

    The final population is returned, not just the winner, because Phase 3 needs
    a diverse set of evasive strategies to fold back into the detector's
    training data. One strategy would teach the detector one lesson.
    """
    if epoch is None:
        epoch = detector.epoch

    toolbox, _ = build_toolbox(
        detector, graph, tau, ledger, cfg, fitness_cfg, amount_goal, deadline_hours, epoch
    )

    population = toolbox.population(n=cfg.population)
    history: list[GenerationRecord] = []

    for gen in range(cfg.generations + 1):
        # Evaluate everyone whose fitness is stale. Elites keep theirs, which is
        # the point of elitism -- but note their fitness was measured on
        # different episodes, so the reported best is a slightly optimistic
        # sample. That is why the report also re-scores the winner on fresh
        # episodes at the end.
        stale = [ind for ind in population if not ind.fitness.valid]
        for ind in stale:
            ind.fitness.values = toolbox.evaluate(ind)

        raw = np.array([ind.fitness.values[0] for ind in population], dtype=np.float64)
        best_idx = int(np.argmax(raw))
        best = population[best_idx]
        diag = getattr(best, "diagnostics", {})

        history.append(
            GenerationRecord(
                generation=gen,
                best_fitness=float(raw[best_idx]),
                mean_fitness=float(raw.mean()),
                best_completion=float(diag.get("completion", 0.0)),
                best_evasion=float(diag.get("evasion", 0.0)),
                best_n_transactions=float(diag.get("n_transactions", 0.0)),
                diversity=_population_diversity(population),
                best_genome=list(map(float, best)),
            )
        )

        if progress and (gen % 5 == 0 or gen == cfg.generations):
            print(
                f"  gen {gen:>3}  best={raw[best_idx]:.4f}  mean={raw.mean():.4f}  "
                f"completion={diag.get('completion', 0):.3f}  "
                f"evasion={diag.get('evasion', 0):.3f}  "
                f"txns={diag.get('n_transactions', 0):.0f}  "
                f"diversity={history[-1].diversity:.3f}",
                flush=True,
            )

        if gen == cfg.generations:
            break

        # Elites carry forward untouched.
        elite = list(map(toolbox.clone, tools.selBest(population, cfg.elitism)))

        # Selection on shared fitness, so crowded niches are discounted.
        shared = _shared_fitness(population, raw)
        for ind, value in zip(population, shared):
            ind.fitness.values = (float(value),)
        offspring = list(map(toolbox.clone, toolbox.select(population, cfg.population - cfg.elitism)))
        # Restore raw fitness on the parents; offspring are invalidated below.
        for ind, value in zip(population, raw):
            ind.fitness.values = (float(value),)

        for a, b in zip(offspring[::2], offspring[1::2]):
            if random.random() < cfg.crossover_prob:
                toolbox.mate(a, b)
                del a.fitness.values, b.fitness.values
        for ind in offspring:
            if random.random() < cfg.mutation_prob:
                toolbox.mutate(ind)
                del ind.fitness.values

        # Hard repair after every operator, so the population is always inside
        # the plausibility envelope (see constraints.py).
        for ind in offspring:
            ind[:] = repair_genome(np.asarray(ind, dtype=np.float64)).tolist()

        population = elite + offspring

    best_genome = np.asarray(max(population, key=lambda i: i.fitness.values[0]), dtype=np.float64)
    return best_genome, history, population


def distinct_top(population: list, k: int, min_distance: float = SIGMA_SHARE) -> list[np.ndarray]:
    """The ``k`` best genomes that are mutually at least ``min_distance`` apart.

    Phase 3 feeds these back into the detector's training set. Taking the top-k
    by fitness alone would usually return k near-identical genomes -- the GA
    converges -- and retraining on k copies of one strategy teaches the detector
    almost nothing. Enforcing separation is what makes the feedback informative.
    """
    ordered = sorted(population, key=lambda i: -i.fitness.values[0])
    chosen: list[np.ndarray] = []
    for ind in ordered:
        g = np.asarray(ind, dtype=np.float64)
        if all(distance(g, other) >= min_distance for other in chosen):
            chosen.append(g)
        if len(chosen) >= k:
            break
    return chosen
