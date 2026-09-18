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
The winning genome set `threshold_hug = 0.678`, not 1.0 — every transaction
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
