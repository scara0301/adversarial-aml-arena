"""The Phase 3 gate.

These checks ask whether the *loop and its guards work*. None asks the detector
to win: who is ahead at round R is the experiment's finding, and a gate that
required one side to win would be a result rigged in advance.
"""

from __future__ import annotations

import numpy as np


def gate(res: dict, max_holdout_drop: float = 0.10) -> list[tuple[str, bool, str]]:
    """Return ``(label, passed, detail)`` for each check.

    1. **No collapse.** No round's clean-holdout recall falls more than
       ``max_holdout_drop`` below round 0 -- the guard against the detector being
       dragged toward the adversary and away from real laundering.
    2. **The feedback reaches the detector.** Averaged over rounds, D_{r+1} flags
       more of A_r's champion than D_r did -- it learned the strategy it was shown.
       Failing this means the pipeline is broken (e.g. a feature mismatch), not
       that the adversary is clever.
    3. **The budget policy held.** Every round's alert rate on the clean holdout
       sits on the fixed budget (invariant 3).
    """
    recall = np.array(res["crossplay"]["recall"])
    n_adv = recall.shape[1]
    hold = [d["holdout"]["recall_at_budget"] for d in res["detectors"]]
    budget = res["settings"]["budget"]

    drop = hold[0] - min(hold)
    gains = [recall[j + 1, j] - recall[j, j] for j in range(n_adv)]
    worst_flag_err = max(abs(d["holdout"]["flag_rate"] - budget) for d in res["detectors"])

    return [
        (f"no holdout collapse (largest drop vs round 0 <= {max_holdout_drop:.2f})",
         drop <= max_holdout_drop, f"largest drop {drop:+.3f}  (round 0 {hold[0]:.3f}, min {min(hold):.3f})"),
        ("feedback reaches the detector (mean recall gain on the strategy it was shown > 0.10)",
         float(np.mean(gains)) > 0.10,
         f"mean gain {np.mean(gains):+.3f}  per round {[round(float(g), 3) for g in gains]}"),
        ("alert budget held every round (invariant 3)",
         worst_flag_err < 0.0005, f"worst |flag rate - budget| = {worst_flag_err:.5f}"),
    ]
