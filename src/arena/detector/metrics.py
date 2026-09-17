"""Metrics for a ~0.1% positive rate.

At this prevalence most familiar metrics are actively misleading, so the choice
of metric is a design decision rather than a formality.

**Accuracy is never reported.** A model that predicts "never laundering" scores
99.9%. Quoting it would be worse than quoting nothing.

**ROC-AUC is reported but labelled.** With 1,000 negatives per positive, a large
absolute number of false positives barely moves the false-positive *rate*, so
ROC-AUC looks flattering while the alert queue is useless. It is kept for
comparability with published work, flagged every time it appears.

**Recall at a fixed alert budget is the primary metric.** "Of the laundering that
happened, what fraction did we catch while reviewing only 0.5% of
transactions?" That is the question a bank asks, and -- because the budget pins
the operating point -- it is the only recall number that stays comparable across
arms-race rounds.

**PR-AUC is the threshold-free check.** It is sensitive to the positive class at
low prevalence, so it catches a model that is ranking badly overall even if the
budget-recall happens to look acceptable.

**Per-typology recall** is required by objective 1, which asks for a detector
"capable of recognising known laundering typologies". A single aggregate recall
can hide that the model catches fan-out perfectly and cycles not at all --
exactly the blind spot the adversary will find and exploit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from arena.detector.threshold import flag_rate


def recall_at_threshold(y_true: np.ndarray, scores: np.ndarray, tau: float) -> float:
    """Fraction of true positives scoring at or above ``tau``."""
    pos = y_true == 1
    n_pos = int(pos.sum())
    if n_pos == 0:
        return float("nan")
    return float((scores[pos] >= tau).mean())


def precision_at_threshold(y_true: np.ndarray, scores: np.ndarray, tau: float) -> float:
    """Fraction of alerts that are true positives -- the analyst's hit rate."""
    alerts = scores >= tau
    n_alerts = int(alerts.sum())
    if n_alerts == 0:
        return float("nan")
    return float((y_true[alerts] == 1).mean())


def typology_recall(
    y_true: np.ndarray, scores: np.ndarray, typologies: pd.Series, tau: float
) -> dict[str, float]:
    """Recall at ``tau``, broken out per laundering typology.

    A transaction is counted for its own typology label. Reported per
    transaction rather than per pattern instance, which is the stricter reading:
    catching one leg of a ten-leg fan-out is not the same as catching the
    pattern, and this metric does not pretend otherwise.
    """
    out: dict[str, float] = {}
    labels = typologies.astype("string")
    for name in sorted(labels.dropna().unique()):
        # na_value=False matters: comparing a nullable string column yields NA
        # for benign rows, and an object-dtype mask cannot index a numpy array.
        is_name = (labels == name).to_numpy(dtype=bool, na_value=False)
        mask = is_name & (y_true == 1)
        if mask.sum() == 0:
            continue
        out[str(name)] = float((scores[mask] >= tau).mean())
    return out


def evaluate(
    y_true: np.ndarray,
    scores: np.ndarray,
    tau: float,
    typologies: pd.Series | None = None,
    budget: float | None = None,
) -> dict:
    """Full metric bundle for one evaluation set.

    Returns a JSON-serialisable dict. ``roc_auc`` carries an explicit note
    rather than sitting bare next to the honest metrics.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int((y_true == 1).sum())

    result: dict = {
        "n": int(len(y_true)),
        "n_positive": n_pos,
        "positive_rate": float(y_true.mean()) if len(y_true) else 0.0,
        "threshold": float(tau),
        "alert_budget": budget,
        "flag_rate": flag_rate(scores, tau),
        "recall_at_budget": recall_at_threshold(y_true, scores, tau),
        "precision_at_budget": precision_at_threshold(y_true, scores, tau),
        "n_alerts": int((scores >= tau).sum()),
    }

    # Both AUCs are undefined with a single class present.
    if n_pos > 0 and n_pos < len(y_true):
        result["pr_auc"] = float(average_precision_score(y_true, scores))
        result["roc_auc"] = float(roc_auc_score(y_true, scores))
        result["roc_auc_note"] = (
            "Misleading at this prevalence: with ~1000 negatives per positive, a large "
            "absolute false-positive count barely moves the false-positive rate. "
            "Use recall_at_budget and pr_auc."
        )
    else:
        result["pr_auc"] = float("nan")
        result["roc_auc"] = float("nan")

    if typologies is not None:
        result["typology_recall"] = typology_recall(y_true, scores, typologies, tau)

    return result


def format_report(metrics: dict, title: str = "Detector") -> str:
    """Human-readable summary for stdout and the phase writeups."""
    lines = [
        f"{title}",
        "-" * len(title),
        f"  transactions        {metrics['n']:,}",
        f"  laundering          {metrics['n_positive']:,} ({metrics['positive_rate']:.4%})",
        f"  threshold (tau)     {metrics['threshold']:.6f}",
        f"  alerts raised       {metrics['n_alerts']:,} ({metrics['flag_rate']:.4%} of traffic)",
        "",
        f"  RECALL @ budget     {metrics['recall_at_budget']:.4f}   <-- primary metric",
        f"  PR-AUC              {metrics['pr_auc']:.4f}",
        f"  precision @ budget  {metrics['precision_at_budget']:.4f}",
        f"  ROC-AUC             {metrics['roc_auc']:.4f}   (misleading at this prevalence)",
    ]
    if metrics.get("typology_recall"):
        lines += ["", "  recall by typology"]
        for name, value in sorted(metrics["typology_recall"].items(), key=lambda kv: -kv[1]):
            lines.append(f"    {name:<16}{value:.4f}")
    return "\n".join(lines)
