"""Architecture walkthrough: why the system is shaped the way it is."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    st.title("⚙️ How It Works")

    st.subheader("The crux: one shared featurize()")
    st.markdown(
        """
The detector and the adversary must speak through the same interface, or the
whole exercise is void: an adversary allowed to edit feature vectors
directly isn't evading detection, it's cheating.

```
training:   full ledger  ──▶  AccountGraph.build()  ──▶  featurize()  ──▶  LightGBM
adversary:  sub-ledger    ──▶  frozen.overlay(sub)   ──▶  featurize()  ──▶  same LightGBM
```

Both paths call the identical `featurize()`. The only difference is which
`AccountGraph` they pass in — a graph built fresh over the whole ledger, or a
**frozen snapshot with a copy-on-write overlay**.
"""
    )

    st.subheader("Why the overlay, not a full rebuild")
    st.markdown(
        """
A single GA run is population × generations × episodes ≈ **20,500 candidate
evaluations**. Re-featurizing a 2-million-row background ledger for every
candidate is impossible.

`AccountGraph.build()` does one pass over the background, once per round.
`AccountGraph.overlay(sub_ledger)` then rebuilds state only for the handful
of accounts a candidate actually touches — everything else is shared by
reference. Measured: **4.8 ms per candidate** against a 100k-row background,
which is what makes the GA affordable at all.
"""
    )
    st.info(
        "Batch (training) and overlay (adversary) featurization were measured "
        "to agree exactly — max difference **0.0** across a real sub-ledger. "
        "That equivalence *is* invariant 1."
    )

    st.subheader("Money conservation, by construction")
    st.markdown(
        """
The genome never encodes an absolute amount — only relative quantities: a
split count, a skew, a fraction. The decoder always allocates exactly the
episode's goal amount across however many tranches the genes ask for, and
the final tranche absorbs any float rounding.

The consequence: **any** genome, including a crossover offspring that never
existed in either parent, conserves money exactly. Nothing needs to check or
repair it after the fact — there is no absolute quantity anywhere for an
operator to corrupt.
"""
    )

    st.subheader("Guarding against the GA's own cleverness")
    st.markdown(
        """
A genetic algorithm is an adversarial search over *your own scoring
function*. Left unguarded, it will find every way to score well without
doing the task — and during this project's development, it did exactly that
once: with zero delay between transactions, the decoder emitted twelve
transfers at one identical timestamp. Because the as-of feature queries
exclude ties, none of those transactions could see each other, every burst
feature read zero, and the detector's recall on that "strategy" was exactly
**0.000** — not because the adversary was clever, but because it had found a
hole in the simulation.

The fix was a minimum one-minute gap between transfers from the same
account, now enforced by the decoder and checked by a regression test. The
general lesson: **when an adversarial search reports a perfect score, assume
it found a bug, not a breakthrough.**
"""
    )

    st.subheader("The evolved strategy's real insight")
    st.markdown(
        """
The winning genome set `threshold_hug` to about 0.68 (0.678 and 0.675 in two independent runs), not 1.0 — every transaction
in the winning strategy sits *above* the $10,000 reporting threshold, the
single feature the detector leans on hardest.

Full structuring would need 26+ tranches instead of 13, and that many
transactions in a short window trips the detector's *burst* features
instead. Against a graph-aware detector, evading one feature family by
triggering another isn't a win. The GA found that sitting in the fat tail of
the **benign** amount distribution, and keeping the transaction count low,
beats structuring outright. See it for yourself on **Try the Adversary
Yourself** — push `threshold_hug` to 1.0 and watch the transaction count (and
the flag rate) climb.
"""
    )

    st.subheader("The closed loop, and the guards that keep it honest")
    st.markdown(
        """
Phase 3 retrains the detector on what the adversary found, round after round.
That is exactly where an arms-race experiment usually fools itself, so every
step has a guard:

- **A cap.** Adversary rows are at most 20% of the *original* positive count.
  Flood the training set with them and the detector is dragged toward the
  adversary and away from real laundering.
- **Always retrain from the original set plus the pool**, never from the
  previous round's already-augmented set. Augmenting an augmented set
  compounds drift every round.
- **Equal slices per round**, counted in rows, so learning a new evasion never
  costs the detector an old one. (A first version counted operations instead of
  rows; operations range from 2 to 100+ rows, so one round could starve another
  — caught by a test, not by luck.)
- **Time windows.** The detector trains on operations from the training period,
  validates on the validation period, and every adversary is *judged* on episodes
  from a later period still. Same reason every split in this project is
  chronological.
- **Only strategies that completed the goal** are fed back. A strategy that
  never moved the money is a failed attempt, not a discovered evasion.
- **Fixed episodes and a fixed decode seed** for every comparison, so two cells of
  the cross-play matrix differ because the *players* differ, never because of
  the dice.
- **A runtime check on the holdout.** It is fingerprinted before the run and
  re-checked after; the run fails loudly if a single value changed.

The phase gate checks that these guards work. It never asks the detector to win
— who is ahead at the end is the finding, and a gate demanding a winner would be
rigged in advance.
"""
    )
