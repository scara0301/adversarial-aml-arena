"""Figures for the arms-race result.

Pure functions over the JSON-shaped dict ``ArenaResult.to_json()`` produces, so
the same plots can be redrawn from ``reports/phase3_metrics.json`` without
re-running anything.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from arena.adversary.genome import GENES

_RED, _BLUE, _GREEN, _GREY, _PURPLE = "#c53030", "#2b6cb0", "#2f855a", "#718096", "#805ad5"


def _rounds(res: dict) -> tuple[list[int], list[int]]:
    return [d["round"] for d in res["detectors"]], [a["round"] for a in res["adversaries"]]


def plot_arms_race(res: dict, out: Path) -> None:
    """The headline curve: the detector's recall against the adversary, and its
    recall on clean data, round by round.

    Two panels, because they answer different questions. Top: *can the adversary
    still win?* Bottom: *what is the detector paying for its robustness?* A
    detector that got robust by flagging everything would look great on the top
    panel and collapse on the bottom one.
    """
    d_rounds, a_rounds = _rounds(res)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    own = [a["champion_vs_own"] for a in res["adversaries"]]
    ax1.plot(a_rounds, [s["detector_recall"] for s in own], "o-", lw=2.5, color=_RED,
             label="detector recall vs. its contemporary adversary (A_r vs D_r)")
    if all("ci" in s for s in own):   # 95% bootstrap interval over episodes
        lo = np.array([s["ci"]["detector_recall"][0] for s in own])
        hi = np.array([s["ci"]["detector_recall"][1] for s in own])
        ax1.fill_between(a_rounds, lo, hi, color=_RED, alpha=0.15, label="95% interval (bootstrap over episodes)")
    ax1.plot(a_rounds, [1 - s["evasion"] for s in own], "s--", lw=1.5, color=_PURPLE,
             label="value flagged = 1 - evasion")
    ax1.plot(a_rounds, [s["success_rate"] for s in own], "^-", lw=1.5, color=_GREEN,
             label="adversary success rate")
    ax1.plot(d_rounds, [d["naive"]["detector_recall"] for d in res["detectors"]], ":", lw=1.5,
             color=_GREY, label="recall vs. the fixed naive strategy")
    ax1.plot(d_rounds, [d["random"]["detector_recall"] for d in res["detectors"]], "-.", lw=1.5,
             color=_BLUE, label="recall vs. unseen random strategies")
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("detector recall / rate")
    ax1.set_title("The arms race: every number is on held-out episodes from a later period")
    ax1.legend(fontsize=8, loc="best")
    ax1.grid(alpha=0.25)

    hold = [d["holdout"]["recall_at_budget"] for d in res["detectors"]]
    ax2.plot(d_rounds, hold, "o-", lw=2.5, color=_BLUE, label="recall @ 0.5% alert budget")
    ax2.plot(d_rounds, [d["holdout"]["pr_auc"] for d in res["detectors"]], "s--", lw=1.5,
             color=_GREEN, label="PR-AUC")
    ax2.axhline(hold[0], color=_GREY, lw=0.8, ls=":", label=f"round 0 recall ({hold[0]:.3f})")
    ax2.set_ylim(0, 1.0)
    ax2.set_xlabel("round r  (detector D_r)")
    ax2.set_ylabel("score on the clean holdout")
    ax2.set_title("The cost: performance on the frozen clean holdout (never sees adversary data)")
    ax2.set_xticks(d_rounds)
    ax2.legend(fontsize=8, loc="best")
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_crossplay(res: dict, out: Path) -> None:
    """Detector (rows) against adversary generation (columns), as detector recall.

    Read it by region: the diagonal is the live fight, the cells beneath it are
    memory, the cells above it are progress. See orchestrator.py.
    """
    M = np.array(res["crossplay"]["recall"])
    n_d, n_a = M.shape
    fig, ax = plt.subplots(figsize=(1.1 * n_a + 3, 0.8 * n_d + 2.5))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for i in range(n_d):
        for j in range(n_a):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=10,
                    fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(n_a), [f"A_{j}" for j in range(n_a)])
    ax.set_yticks(range(n_d), [f"D_{i}" for i in range(n_d)])
    ax.set_xlabel("adversary generation (champion)")
    ax.set_ylabel("detector generation")
    ax.set_title("Cross-play: detector recall. Green = detector wins, red = adversary wins")
    fig.colorbar(im, ax=ax, label="detector recall")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_typology(res: dict, out: Path) -> None:
    """Holdout recall per typology, per round -- the catastrophic-forgetting guard.

    If feedback teaches the detector one shape by making it forget another, it
    shows up here as a typology line falling away from the rest.
    """
    d_rounds, _ = _rounds(res)
    names = sorted({k for d in res["detectors"] for k in d["holdout"]["typology_recall"]})
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for name in names:
        ax.plot(d_rounds, [d["holdout"]["typology_recall"].get(name, np.nan) for d in res["detectors"]],
                "o-", lw=1.8, label=name)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(d_rounds)
    ax.set_xlabel("round r  (detector D_r)")
    ax.set_ylabel("clean-holdout recall @ alert budget")
    ax.set_title("Per-typology recall on the clean holdout, by round")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_gene_drift(res: dict, out: Path) -> None:
    """How the champion's genome moves from round to round, each gene scaled to
    its own range -- *what the adversary changes in response to the detector*."""
    _, a_rounds = _rounds(res)
    names = [g.name for g in GENES if g.name != "typology_template"]
    span = {g.name: (g.low, g.high) for g in GENES}
    M = np.array([
        [(a["champion"][n] - span[n][0]) / (span[n][1] - span[n][0]) for n in names]
        for a in res["adversaries"]
    ]).T
    fig, ax = plt.subplots(figsize=(1.0 * len(a_rounds) + 4.5, 0.42 * len(names) + 2))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = res["adversaries"][j]["champion"][names[i]]
            ax.text(j, i, f"{v:.2f}" if isinstance(v, float) else str(v), ha="center", va="center",
                    fontsize=8, color="white" if M[i, j] < 0.55 else "black")
    ax.set_xticks(range(len(a_rounds)), [f"A_{j}" for j in a_rounds])
    ax.set_yticks(range(len(names)), names)
    templates = [a["champion"].get("typology_template_name", "?") for a in res["adversaries"]]
    ax.set_title("Champion genome by round (colour = position within the gene's range)\n"
                 "template: " + " -> ".join(templates), fontsize=9)
    fig.colorbar(im, ax=ax, label="position in range")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def print_summary(res: dict) -> None:
    """The table a reader actually wants, plus the cross-play matrix."""
    print("=" * 78)
    print("ARMS-RACE SUMMARY")
    print("=" * 78)
    ev = res.get("evaluation", {})
    print(f"measured on {res['settings']['eval_episodes']} held-out episodes"
          + (f"; intervals are {ev['ci']}" if ev else ""))
    print(f"{'round':>5} | {'holdout recall':>14} {'PR-AUC':>7} | {'D_r vs A_r':>10} {'success':>8} "
          f"| {'vs naive':>8} {'vs random':>9}")
    for r, d in enumerate(res["detectors"]):
        own = res["adversaries"][r]["champion_vs_own"] if r < len(res["adversaries"]) else None
        ci = ""
        if own and "ci" in own:
            lo, hi = own["ci"]["detector_recall"]
            ci = f"  [{lo:.2f}-{hi:.2f}]"
        print(f"{r:>5} | {d['holdout']['recall_at_budget']:>14.3f} {d['holdout']['pr_auc']:>7.3f} | "
              f"{(own['detector_recall'] if own else float('nan')):>10.3f} "
              f"{(own['success_rate'] if own else float('nan')):>8.2f} | "
              f"{d['naive']['detector_recall']:>8.3f} {d['random']['detector_recall']:>9.3f}{ci}")
    M = np.array(res["crossplay"]["recall"])
    print("\ncross-play: detector recall, rows D_i, columns A_j")
    print("      " + "".join(f"{'A_' + str(j):>7}" for j in range(M.shape[1])))
    for i, row in enumerate(M):
        print(f"{'D_' + str(i):>5} " + "".join(f"{v:>7.2f}" for v in row))


def plot_all(res: dict, reports: Path) -> list[Path]:
    reports.mkdir(parents=True, exist_ok=True)
    paths = {
        "phase3_arms_race.png": plot_arms_race,
        "phase3_crossplay.png": plot_crossplay,
        "phase3_typology_recall.png": plot_typology,
        "phase3_gene_drift.png": plot_gene_drift,
    }
    for name, fn in paths.items():
        fn(res, reports / name)
    return [reports / n for n in paths]
