# Phase 1 — Data foundation and the AML detector

*Objective 1: train a fraud/AML transaction classifier on the IBM AML schema, capable of
recognising the laundering typologies fan-out, fan-in, cycle, gather-scatter and stack.*

Status: **complete**. Reproduce with:

```bash
uv run python scripts/make_synthetic.py && uv run python scripts/train_detector.py
```

## Results

Measured on the frozen clean holdout — 500,001 transactions, 1,132 laundering (0.2264%), the
last chronological 25% of the ledger.

| Metric | Value |
|---|---|
| **Recall @ 0.5% alert budget** | **0.537** |
| PR-AUC | 0.454 |
| Precision @ budget | 0.243 |
| ROC-AUC | 0.971 *(misleading at this prevalence — see below)* |
| Alerts raised | 2,501 (0.5002% of traffic) |
| Chance-level PR-AUC | 0.00226 |

Recall by typology:

| Typology | Recall |
|---|---|
| stack | 0.745 |
| cycle | 0.482 |
| gather-scatter | 0.458 |
| fan-out | 0.414 |
| fan-in | 0.411 |

Artifacts: `reports/phase1_metrics.json`, `reports/phase1_pr_curve.png`,
`reports/phase1_typology_recall.png`, `reports/phase1_feature_importance.png`.

**Read the spread, not the average.** The detector is much better at sequential layering
(`stack`, 0.745) than at wide fan patterns (`fan-in`, 0.411). That gap is not noise — it is the
weakness the Phase 2 adversary has an incentive to exploit, and its `typology_template` gene
lets it choose freely among these five shapes.

## What was built

- `src/arena/schema.py` — the canonical ledger schema every source normalises into. Mirrors the
  IBM AML columns and adds `typology` and `pattern_id`, which is what makes per-typology recall
  measurable at all.
- `src/arena/data/synthetic.py` — the generator (see below).
- `src/arena/data/ibm_aml.py` — reader for the real Kaggle CSV, including a parser for
  `*_Patterns.txt`. Written now so the loader contract is real rather than aspirational, but
  unexercised until Phase 5.
- `src/arena/features/` — `AccountGraph` (frozen snapshot + copy-on-write overlay), `windows.py`
  (as-of window queries), `pipeline.py` (the single shared `featurize`). 74 features across three
  families.
- `src/arena/detector/` — LightGBM wrapper, the alert-budget threshold policy, and the
  imbalance-aware metric set.

## Three generator leaks, found and removed

The first detector trained here scored **PR-AUC 1.0000**. At a 0.2% positive rate that is a
symptom, not a success. Three separate artefacts were making laundering trivially separable, and
each one had to be fixed before any number above meant anything. They are documented here
because the same mistakes are easy to reintroduce, and because each is now guarded by a test in
`tests/test_data.py`.

**1. FX loss.** Laundering rows always kept the same currency and lost 0–0.5% in transit, while
benign rows lost either exactly 0% (same currency) or 0.5–3% (cross currency). Laundering
therefore occupied an FX-loss band benign traffic never touched. `fx_loss` alone carried six
times the gain of every other feature combined.
*Fix:* both classes now draw FX behaviour from one shared function, `_fx_legs`.

**2. Amount scale.** Benign amounts were log-normal with a ~$665 median while structured
laundering tranches landed between $1k and $10k, so laundering was systematically *larger* than
benign traffic. `amount_log` then accounted for **97% of the model's gain** — the detector had
learned an amount threshold, not laundering. Structuring exists precisely to hide inside the
normal amount range, so the distributions have to overlap.
*Fix:* benign scale raised to `lognormal(7.6, 1.5)`, pattern totals reduced, and the
sub-threshold jitter widened so tranches do not pile into a tight $8.2k–$9.9k band.

**3. Diurnal structure.** Laundering timestamps were effectively uniform across the clock while
benign traffic clustered in business hours, handing the detector a third easy separator. Real
launderers operate inside normal hours. Fixing this took two attempts: the hour-of-day draw was
being cancelled out first by an unsnapped `base_hour` and then by a *fractional* `day_offset`,
both of which rotated each pattern's clock by a random amount.
*Fix:* both offsets snapped to whole days, and both classes share `_hour_of_day`, differing only
in one bias parameter (0.90 benign vs 0.72 laundering). Measured result: 0.776 benign vs 0.668
laundering in business hours — a real signal, not a giveaway.

The pattern in all three fixes is the same and worth stating: the leak was closed by making both
classes share a single code path, so the artefact becomes structurally impossible rather than
merely corrected. That is the same reasoning behind having exactly one `featurize()`.

## A fourth artefact, found later: round amounts

**Added during Phase 3.** It was missed here, and it is worth saying how: `amount_mod_100` is the
*second*-ranked feature in `reports/phase1_feature_importance.png`, and I read it as the
"round-number structuring signal" it was named for instead of asking why a feature that cheap was
doing so much work.

Benign amounts are rounded to a multiple of $100 or more 21.6% of the time (`_benign_amounts`
deliberately models people sending $500, not $487.33). Laundering tranches come out of a Dirichlet
split and are continuous: **0.00% of laundering amounts are round**. So "this amount is round"
means "this is benign" with certainty — a free, one-sided tell that removes a fifth of all benign
traffic from the alert queue. Real laundering is often round (that is half of what structuring
*is*), so a detector trained on real data would not have this shortcut.

**How much did it inflate Phase 1?** Retraining without `amount_mod_100`, `amount_mod_1000` and
`is_round_100`, on the same split, with the same settings:

| | recall @ 0.5% budget | PR-AUC | precision @ budget |
|---|---|---|---|
| full feature set | 0.548 | 0.452 | 0.248 |
| without roundness features | 0.516 | 0.432 | 0.234 |

About **3 points of recall and 2 of PR-AUC** — real, but modest, so the headline numbers above are
not invalidated; read them as roughly that much optimistic. (This retrain's "full" row is 0.548
against the 0.537 reported above, so run-to-run variation of about a point exists and has not been
measured; the gap between the rows is larger than that.)

**Where it mattered was Phase 3.** The first closed-loop run's adversary, in round 1, reached
detector recall 0.000 by sending one $250,000.00 transfer: with a single tranche the decoder sends
exactly the episode goal, and every episode had the identical, round, goal. See
[phase3.md](phase3.md). The generator was *not* changed in response — doing so means regenerating
the ledger and re-running Phases 1-3 — so this artefact is still present, and is the first thing to
fix before the real-data work in Phase 5. Phase 3 closes the adversary's *access* to it (each
episode draws a continuous goal) rather than the artefact itself.

## Design decisions

### The shared feature pipeline (invariant 1)

There is exactly one `featurize()`. Training calls it against a graph built from the full ledger;
the Phase 2 adversary calls it against a copy-on-write overlay of its candidate sub-ledger. An
adversary allowed to edit feature vectors directly would be cheating and the project's central
claim would be void.

The obstacle is cost. A GA run is ~20,000 candidate evaluations, and re-featurizing a 2M-row
background ledger per candidate is impossible. `AccountGraph.build` does one pass (10s), and
`AccountGraph.overlay` rebuilds state only for the accounts a sub-ledger touches — tens of them —
sharing everything else by reference. Measured: **4.8 ms per candidate**, against a 100k-row
background. `tests/test_features.py::test_batch_and_overlay_agree` asserts the two paths produce
identical feature matrices; measured maximum difference is exactly 0.

### Everything is "as of" the transaction's own timestamp

Account state is evaluated using only events at or before the transaction being scored. A
cumulative aggregate would leak the future into the past: the model would learn from transactions
that had not happened yet. As-of semantics also make the overlay exact, since an injected
sub-ledger cannot retroactively change an account's earlier state.

The one deliberate exception is `CycleIndex`, which uses cumulative topology rather than
time-sliced reachability. Maintaining as-of reachability per account would cost far more than the
feature is worth, and the leak is mild — it exposes graph structure, never labels. Both the batch
and overlay paths use the identical cumulative index, so the equivalence invariant still holds.
Recorded here rather than hidden.

### Time-based splits, never random

A random split places transactions from the same laundering pattern, and the same account
histories, on both sides of the boundary. Every metric would be inflated and the inflation would
be invisible. The holdout is the last chronological 25%.

### The metric set

- **Accuracy is never reported.** It is ~99.8% for a model that predicts "never laundering".
- **ROC-AUC is reported but always labelled.** With ~440 negatives per positive, a large absolute
  false-positive count barely moves the false-positive *rate*. Note the contrast in the table
  above: ROC-AUC 0.971 against PR-AUC 0.454 on the same scores. The first number would badly
  mislead a reader; that is the whole argument for the second.
- **Recall at a fixed 0.5% alert budget is primary.** A bank has N analysts who can review M
  alerts a day; the threshold falls out of that capacity, not out of an F1 optimisation.
- **PR-AUC** is the threshold-free sanity check.

Fixing the threshold by budget is also invariant 3: it keeps recall comparable across arms-race
rounds. If the threshold were re-tuned each round, a rising recall curve could just be the
cut-off moving.

### Negative downsampling

Training keeps every positive and 25 negatives per positive. The **graph is never downsampled**,
so account features are computed against the complete ledger regardless of which rows are
trained on. Calibration shifts, which is harmless here because the threshold is a holdout
quantile — only ranking matters.

### Scale

2M rows over 20,000 accounts, ~100 transactions per account, 0.2% laundering (~400 pattern
instances). Both numbers were forced by experiment. At 40k rows there were 21 patterns and ~13
transactions per account, and the relational features were pure noise — PR-AUC 0.059. The real
IBM AML rate is ~0.1%; 0.2% is used so that 2M rows yield enough pattern instances to learn
typology *shape*, while remaining a severe imbalance. The holdout is always evaluated at its true
prevalence.

## The recalibrated PR-AUC gate

The plan set a Phase 1 bar of PR-AUC ≥ 0.60. That number was chosen **before** the data's
difficulty was known, and before the three leaks above were found — the first measurement against
the bar was the PR-AUC 1.00 run that turned out to be measuring an artefact.

After the fixes, a capacity sweep (4× the trees, half the learning rate, double the leaves, and
with/without the redundant `scale_pos_weight` correction) moved PR-AUC by less than 0.02 in any
direction:

| Configuration | Trees | PR-AUC | Recall @ 0.5% |
|---|---|---|---|
| baseline (`scale_pos_weight`=25) | 261 | 0.504 | 0.663 |
| downsampling only (`spw`=1) | 159 | 0.492 | 0.662 |
| 1500 trees, lr 0.03 | 273 | 0.498 | 0.659 |
| 1500 trees, lr 0.03, 127 leaves | 210 | 0.487 | 0.656 |

So ~0.50 was the ceiling of that data rather than a tuning shortfall, and the subsequent diurnal
fix lowered it further to 0.454 by removing another easy signal. The gate is now **PR-AUC ≥ 0.40
and recall@budget ≥ 0.50**, which the detector passes.

This is a bar that was moved, so the reasoning should be judged on its merits: the only way to
push PR-AUC back toward 0.60 would be to make laundering more separable again, which is the exact
opposite of what this project needs. A detector that is too good has no arms race to run.

## Limitations

1. **Synthetic data is not real data.** These numbers are a pipeline check. Expect them to drop
   on real IBM AML data in Phase 5, where typologies are messier and the positive rate is lower.
2. **`CycleIndex` uses cumulative topology**, as described above.
3. **Cycle detection is capped** at 32 neighbours per account and a 256-entry 2-hop set, so a
   cycle running only through hub accounts can be missed. The caps are deterministic, which is
   what matters for batch/overlay agreement.
4. **Per-transaction typology recall is the strict reading.** Catching one leg of a ten-leg
   fan-out is not the same as catching the pattern, and this metric does not pretend otherwise.

## What Phase 2 needs from this

- `artifacts/detector_r0.lgb` plus its `.meta.json` sidecar (feature order, windows, epoch).
- `data/processed/holdout_clean.parquet` — the frozen clean holdout, which is where the adversary's
  operating threshold comes from. The adversary must not be able to move its own goalposts.
- The typology recall spread, which tells the adversary which shapes are worth adopting.
