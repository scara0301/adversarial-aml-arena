"""Decision threshold policy.

**Invariant 3.** The threshold is set by a fixed *policy*, never tuned freely.

The policy is a fixed **alert budget**: flag the top ``budget`` fraction of
transactions by score, and no more. Two reasons, and the second is the one that
matters for this project.

First, it is what real AML teams actually do. A bank does not choose a
probability cut-off; it has N analysts who can review M alerts a day, and the
threshold falls out of that capacity. Optimising a threshold on F1 or Youden's J
produces a number nobody in the field would use.

Second, and critically: it makes recall *comparable across arms-race rounds*. If
the threshold were re-optimised each round, the detector's recall could rise or
fall purely because the cut-off moved, and the arms-race curve -- the project's
headline result -- would be measuring threshold drift rather than learning. The
budget pins the operating point so that a change in recall means a real change
in detection ability.

The threshold is always computed on the **frozen clean holdout** (invariant 4),
never on data the adversary has contaminated. Otherwise the adversary could move
the operating point simply by flooding the evaluation set.
"""

from __future__ import annotations

import numpy as np


def threshold_for_budget(scores: np.ndarray, budget: float = 0.005) -> float:
    """Score cut-off that flags exactly ``budget`` of ``scores``.

    Args:
        scores: Detector scores on the frozen clean holdout.
        budget: Fraction of transactions the institution can afford to review.
            Default 0.005 (0.5%).

    Returns:
        The threshold tau. Scores ``>= tau`` are alerts.
    """
    if not 0.0 < budget < 1.0:
        raise ValueError(f"budget must be in (0, 1), got {budget}")
    if scores.size == 0:
        raise ValueError("cannot derive a threshold from an empty score array")

    # The (1 - budget) quantile: everything above it is inside the budget.
    return float(np.quantile(scores, 1.0 - budget, method="higher"))


def flag_rate(scores: np.ndarray, tau: float) -> float:
    """Fraction of transactions the threshold would alert on.

    Reported alongside recall as a sanity check: on the holdout it should land
    on the budget, and on an adversary population it is free to be anything.
    """
    return 0.0 if scores.size == 0 else float((scores >= tau).mean())
