"""Re-measure a finished arms race with more episodes and error bars.

The expensive part of Phase 3 is the *search* (a GA per round); the cheap part is
the *measurement* (decode a strategy, score it). They have different noise
budgets, and the first closed-loop runs showed why they should be separable:

A strategy that sends one or two transactions is close to a coin flip each
episode -- the detector flags that transfer or it does not, with nothing in
between. Averaged over 30 episodes that leaves a standard error near 0.09, so
two honest samples of the *same* champion gave detector recall readings 0.25
apart. A round-to-round change of that size is noise, and the search itself is
worse: scoring thousands of coin-flip genomes on 5 episodes each guarantees some
score 5-for-5 by luck, and elitism keeps that lucky fitness forever. The GA's
"best fitness 0.999" was a winner's curse; the held-out measurement is the
number to believe, and it deserves a confidence interval.

So this module re-scores every saved (detector, champion) pair on a larger fixed
episode set -- with a percentile bootstrap over episodes -- *without* re-running
any GA. Same detectors, same champions, same decode seed; only the sample size
changes.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from arena.adversary.fitness import EpisodeSampler, FitnessConfig
from arena.adversary.ga import naive_genome
from arena.adversary.genome import GENES, random_genome
from arena.detector.model import Detector
from arena.features import AccountGraph
from arena.loop.evaluation import bootstrap_ci, episode_outcomes, summarize


def genome_from_dict(d: dict) -> np.ndarray:
    return np.array([d[g.name] for g in GENES], dtype=np.float64)


def reevaluate(
    res: dict,
    ledger: pd.DataFrame,
    detectors: list[Detector],
    fitness_cfg: FitnessConfig,
    *,
    n_episodes: int = 200,
    n_boot: int = 1000,
) -> dict:
    """Return a copy of ``res`` with every measurement redone on ``n_episodes``.

    Overwrites, in the copy: each detector's ``naive`` / ``random`` probes, each
    adversary's ``champion_vs_own`` / ``pool_vs_own`` (with a bootstrap 95% CI on
    the champion), and the cross-play matrices (with CI bounds).
    """
    out = copy.deepcopy(res)
    s = res["settings"]
    epoch = detectors[0].epoch
    graph = AccountGraph.build(ledger, epoch=epoch)
    taus = [d["tau"] for d in res["detectors"]]

    goal = res["episode"]["amount_goal"]
    goal = tuple(goal) if isinstance(goal, list) else goal
    deadline = res["episode"]["deadline_hours"]
    w0, w1 = res["windows_hours"]["eval"]
    eval_eps = EpisodeSampler(ledger, epoch).sample(
        n_episodes, np.random.default_rng(s["seed"] + 2), goal, deadline,
        start_range=(w0, w1 - deadline),
    )
    decode_seed = s["seed"] + 7
    boot_rng = np.random.default_rng(s["seed"] + 3)

    champions = [genome_from_dict(a["champion"]) for a in res["adversaries"]]
    pools = [[genome_from_dict(g) for g in a["strategies"]] or [champions[j]]
             for j, a in enumerate(res["adversaries"])]
    naive = naive_genome()
    random_pool = [random_genome(np.random.default_rng(s["seed"] + 100 + i))
                   for i in range(s["n_random_probes"])]

    n_d, n_a = len(detectors), len(champions)
    recall = np.zeros((n_d, n_a)); evasion = np.zeros_like(recall)
    r_lo = np.zeros_like(recall); r_hi = np.zeros_like(recall)

    for i, D in enumerate(detectors):
        tau = taus[i]
        o_naive = episode_outcomes(naive, eval_eps, D, graph, tau, fitness_cfg, decode_seed)
        out["detectors"][i]["naive"] = summarize(o_naive).as_dict()
        out["detectors"][i]["naive"]["ci"] = {k: list(v) for k, v in bootstrap_ci(o_naive, boot_rng, n_boot).items()}

        rand = [summarize(episode_outcomes(g, eval_eps, D, graph, tau, fitness_cfg, decode_seed)).as_dict()
                for g in random_pool]
        out["detectors"][i]["random"] = {k: float(np.mean([r[k] for r in rand])) for k in rand[0]}

        for j, g in enumerate(champions):
            o = episode_outcomes(g, eval_eps, D, graph, tau, fitness_cfg, decode_seed)
            sm, ci = summarize(o), bootstrap_ci(o, boot_rng, n_boot)
            recall[i, j], evasion[i, j] = sm.detector_recall, sm.evasion
            r_lo[i, j], r_hi[i, j] = ci["detector_recall"]
            if i == j:
                own = sm.as_dict()
                own["ci"] = {k: list(v) for k, v in ci.items()}
                out["adversaries"][j]["champion_vs_own"] = own
                prs = [summarize(episode_outcomes(p, eval_eps, D, graph, tau, fitness_cfg, decode_seed)).as_dict()
                       for p in pools[j]]
                out["adversaries"][j]["pool_vs_own"] = {k: float(np.mean([r[k] for r in prs])) for k in prs[0]}

    out["crossplay"] = {
        "recall": recall.tolist(), "evasion": evasion.tolist(),
        "recall_ci_low": r_lo.tolist(), "recall_ci_high": r_hi.tolist(),
    }
    out["evaluation"] = {
        "episodes": n_episodes,
        "original_episodes": s["eval_episodes"],
        "bootstrap_resamples": n_boot,
        "ci": "95% percentile bootstrap over episodes",
        "note": "Re-measured from the saved detectors and champions; no GA was re-run.",
    }
    out["settings"]["eval_episodes"] = n_episodes
    return out
