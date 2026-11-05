# Adversarial Fraud & AML Detection Arena — Final Report

**Batch A5** · Guide: **Dr. B R Vatsala**, Associate Professor

| USN | Name |
|---|---|
| 4NI24CS022 | Anirudh Krishna |
| 4NI24CS067 | Golla Venkata Aditya |
| 4NI24CS068 | Gourav R |
| 4NI24CS010 | Akshay MP |

---

## Abstract

Fraud and anti-money-laundering (AML) detectors are usually validated once, against a fixed dataset
of laundering patterns that were known when the model was trained. A real launderer adapts. This
project builds a closed-loop arena in which a gradient-boosted AML detector faces a genetic-algorithm
adversary that evolves laundering strategies to evade it. The adversary's discoveries are fed back
into the detector's training data, and the two are retrained against each other round by round, so
robustness is measured as a curve against an adapting opponent rather than as one static number.

The system was built and validated on a synthetic transaction world and then run, unchanged, on the
public IBM AML benchmark (HI-Small, 5.08M transactions). On IBM data the detector reaches **recall
0.668 at a 0.5% alert budget** (PR-AUC 0.424, 360× chance) on a frozen, chronologically held-out
set. Against an adaptive opponent it is far weaker: on IBM data even a hand-written laundering strategy goes unflagged, because the detector largely learned that laundering is paid by ACH, and five rounds of retraining on the adversary's strategies did not close the gap. On synthetic data, five rounds of retraining raised the detector's recall
against its contemporary adversary from 0.125 to 0.510 at a cost of under three points of clean
recall, but the adversary still won most encounters. Detectors trained on one world catch
essentially none of the other world's laundering, which is the clearest evidence that a detector's
measured accuracy describes its dataset as much as its ability.

Equally important is what the arena found in the experiment itself. Every time the adversary
reported a near-perfect evasion, the cause was a flaw in the simulation or the data rather than a
genuine strategy: generator leaks, a round-number artefact, a decoder fingerprint, a money-creating
currency path. Each was found by reading the adversary's decoded transactions, and each was fixed
structurally. An adversary turns out to be an effective auditor of the environment it is placed in.

---

## 1. Introduction

### 1.1 Problem

Money laundering is the movement of illegally obtained money through the financial system so that
its origin is hidden. Banks are required to monitor transactions and report suspicious activity, and
machine-learning detectors are increasingly used to rank transactions for analysts to review.

Such detectors face two difficulties that ordinary classification does not:

1. **Extreme imbalance.** Roughly one transaction in a thousand is laundering. A model that predicts
   "not laundering" for everything is 99.9% accurate and useless, so accuracy is meaningless here.
2. **An adaptive opponent.** Launderers change their behaviour in response to detection. A detector
   that scores well on last year's patterns says little about next year's.

The standard evaluation — train on a labelled dataset, report a score on a held-out portion of the
same dataset — addresses neither.

### 1.2 Approach

This project replaces the static evaluation with an arms race:

```
detector D_r  --evolve-->  adversary A_r  --feed back-->  detector D_(r+1)  --> ...
```

At each round a genetic algorithm evolves laundering strategies that evade the current detector
while still delivering the money; the most evasive strategies are added to the detector's training
data; the detector is retrained; and both are measured on fixed, held-out material. The output is a
set of curves and matrices that show whether the detector is getting harder to fool, whether it
forgets earlier lessons, and what each improvement costs on ordinary laundering.

### 1.3 Objectives (from the synopsis)

| # | Objective | Where it is met |
|---|---|---|
| 1 | Train a fraud/AML classifier on the IBM AML dataset that recognises fan-out, fan-in, cycle and gather-scatter typologies | Phase 1 (synthetic), Phase 5 (IBM) |
| 2 | Build a GA adversary whose genome encodes structuring, timing delays and intermediate-account routing, constrained to plausible behaviour | Phase 2 |
| 3 | Define a fitness function balancing evasion against completion of the laundering goal, and evolve strategies over generations | Phase 2 |
| 4 | Feed evasive strategies back into the detector, retrain over multiple rounds, measure detector recall and adversary success | Phase 3 (synthetic), Phase 5 (IBM) |
| 5 | Build a dashboard showing transaction flows, evolved strategies and the detector's response round by round | Phase 4 |

---

## 2. System design

### 2.1 Architecture

```
            ┌───────────────────────────── data ─────────────────────────────┐
            │ synthetic generator  ·  IBM AML loader (HI-Small)   → canonical │
            │                                                       ledger    │
            └──────────────────────────────┬─────────────────────────────────┘
                                           │
                 AccountGraph (frozen snapshot of the background, built once)
                                           │
          ┌────────────────────────────────┼─────────────────────────────────┐
          │                                │                                 │
   featurize()  ◄── one shared path ──►  featurize() on a copy-on-write overlay
   (training rows)                         (adversary's decoded transactions)
          │                                │
   LightGBM detector ── score ──►  GA fitness  ◄── decoder ◄── 12-gene genome
          │                                                       (DEAP)
   alert-budget threshold (fixed policy)
          │
   closed loop: pool (capped, sliced, time-windowed) → retrain → cross-play matrix
          │
   reports/*.json + figures  →  Streamlit dashboard
```

### 2.2 The four invariants

These are not style choices. Each is the difference between a real result and a meaningless one,
and each is enforced by an automated test.

1. **The adversary emits transactions, never features.** An adversary that edited feature vectors
   directly would be cheating. There is exactly one `featurize()`, used both to build training data
   and to score the adversary's decoded transactions. A test checks the batch path and the overlay
   path produce identical features (maximum difference 0.0).
2. **Money conservation is a property of the decoder, not a repair step.** The genome encodes only
   relative quantities (split counts, weights, fractions); the decoder always allocates exactly the
   goal amount. Any genome, including arbitrary crossover offspring, conserves money by construction.
   Tested on 1,500 random offspring (error ≤ $2×10⁻¹⁰), and on the real-data path in Phase 5.
3. **The decision threshold comes from a fixed policy.** It is the score that flags exactly 0.5% of
   transactions on the frozen clean holdout (an analyst-capacity "alert budget"). If the threshold
   moved between rounds, the arms-race curve would measure threshold drift instead of learning.
4. **The frozen clean holdout never sees adversary data.** It is carved out chronologically before
   the adversary exists, and its fingerprint is verified unchanged at the end of every closed-loop run.

### 2.3 Metrics

- **Recall at a 0.5% alert budget** (primary): of all laundering transactions, the fraction the
  detector flags when it may flag only 0.5% of all traffic.
- **PR-AUC** (average precision): threshold-free ranking quality. Chance level equals the prevalence.
- **ROC-AUC** is reported only with the label *misleading at this prevalence*. Accuracy is never reported.
- **Adversary success rate:** fraction of held-out episodes in which the money arrives and less than
  half of its value is flagged.

All splits are chronological. A random split would let the detector see an account's future while
being tested on its past.

---

## 3. Method

### 3.1 Data

**Synthetic world** (Phases 0-4): 2,000,000 transactions over 20,000 accounts and 180 days, 0.2%
laundering, injected as five typologies (fan-out, fan-in, cycle, gather-scatter, stack). Benign and
laundering rows share every generation process that could otherwise separate them (timing, currency,
amount scale) — a rule adopted after three leaks were found (§5).

**IBM AML HI-Small** (Phase 5): 5,078,345 transactions simulated by IBM's AMLworld generator, with a
patterns file listing 370 laundering attempts. It is the standard public AML benchmark; it is itself
simulated (real labelled bank data cannot be published) but was built independently of this project.
Profiling before training found and removed three problems: 591,212 self-transfers, an 8-day tail in
which background traffic has stopped but laundering continues (715 rows, 652 laundering), and 1,968
positives with no pattern label (kept as `unattributed` rather than forced into a typology). Cleaned:
4,486,418 transactions, 0.101% laundering, 422,648 accounts, 10 days.

### 3.2 Features

74 features per transaction, computed as of the transaction's own timestamp (no future information):

- **Per-transaction (17):** amount (log, low-order digits, distance to the $10,000 reporting
  threshold), hour, weekday, cross-bank, cross-currency, FX loss, payment format and its scrutiny.
- **Relational (57):** for both sender and receiver, counts, volumes and distinct counterparties over
  1-hour, 1-day and 7-day windows; burst ratios; account age; time since previous transaction;
  repeated-pair counts; flow ratios; and 2- to 4-hop cycle indicators.

`AccountGraph` holds a frozen snapshot of the background ledger. Scoring an adversary candidate
creates a copy-on-write overlay that rebuilds state only for the few dozen accounts it touches,
which is what makes ~20,000 evaluations per GA run feasible.

### 3.3 Detector

LightGBM gradient-boosted trees (up to 400 trees, 63 leaves, learning rate 0.05), trained on all
positives and 25 negatives per positive, with early stopping on validation PR-AUC. The graph is
never downsampled, only the training rows.

### 3.4 Adversary

A **12-gene genome**: typology template, routing hops, tranches per leg, tranche skew, threshold hug
(how close tranches sit below $10,000), mean delay, delay jitter, business-hours bias, mule-pool size,
mule reuse, payment-format mix, and dormancy at each hop. The decoder turns a genome and an
**episode** (move a goal amount from account S to account D by a deadline, using real background
accounts as mules) into an ordinary sub-ledger of transactions.

**Fitness** = completion^α × evasion^β − γ × cost, set to zero if less than 95% of the money arrives.
Evasion is one minus the share of value the detector flags. Each candidate is averaged over five
episodes so the GA cannot overfit one lucky account. The GA (DEAP) uses population 100,
40 generations, tournament selection, elitism 5.

### 3.5 Closed loop

Five rounds. Each round evolves A_r against D_r, takes up to 8 mutually distinct strategies with
fitness > 0, decodes each on 6 training-period episodes, and retrains D_(r+1) from **the original
training set plus the pool**. Guards, each pinned by a test:

- adversary rows capped at 20% of the original positives, in equal row slices per round, so no
  round's lesson is traded away for another's;
- training-period adversary rows come only from operations that start and finish inside the
  training period (and likewise for validation and evaluation);
- every (detector, adversary) pair is scored on the same fixed episodes with the same decode seed,
  so cells of the cross-play matrix differ only because the players differ;
- final numbers are re-measured on 200 held-out episodes with bootstrap 95% intervals.

---

## 4. Results

### 4.1 Phase 1 — the detector

| frozen clean holdout | synthetic | IBM HI-Small |
|---|---:|---:|
| transactions / laundering | 500,001 / 1,132 | 1,121,605 / 1,324 |
| **recall @ 0.5% budget** | **0.537** | **0.668** |
| PR-AUC | 0.454 | 0.424 |
| precision @ budget | 0.243 | 0.158 |
| ROC-AUC (misleading) | 0.971 | 0.983 |

| recall by typology | fan-out | fan-in | cycle | gather-scatter | stack | unattributed |
|---|---:|---:|---:|---:|---:|---:|
| synthetic | 0.41 | 0.41 | 0.48 | 0.46 | 0.75 | — |
| IBM | 0.91 | 0.95 | 0.94 | 0.96 | 0.86 | 0.26 |

On IBM the labelled typologies are caught well; the 44% of positives without a pattern label are
caught at 0.26.

### 4.2 Phase 2 — the adversary (synthetic)

| strategy | detector recall | money delivered |
|---|---:|---:|
| naive hand-written (3 transfers) | 0.750 | 99.9% |
| random genomes (mean) | 0.748 | — |
| **evolved** (stack, 8 tranches, threshold hug 0.67) | **0.231** | 99.9% |

The GA cut the detector's recall by 52 points while still delivering essentially all of the money.

### 4.3 Phase 3 — the arms race (synthetic)

| round | detector recall vs its contemporary adversary (95% CI) | adversary success | clean holdout recall / PR-AUC |
|---:|---:|---:|---:|
| 0 | 0.125 (0.11-0.14) | 99% | 0.538 / 0.454 |
| 1 | 0.275 (0.22-0.34) | 72% | 0.519 / 0.437 |
| 2 | 0.295 (0.24-0.36) | 70% | 0.529 / 0.441 |
| 3 | 0.365 (0.30-0.44) | 64% | 0.516 / 0.427 |
| 4 | 0.510 (0.48-0.54) | 56% | 0.523 / 0.430 |
| final D_5 | | | 0.509 / 0.421 |

Feedback improved the detector against the strategies it was shown in every round, at a cost of
under 3 points of clean recall. It did not get ahead: the round-4 adversary still succeeds in 56% of
episodes. The cross-play matrix shows the adversary alternating between a multi-hop fan-out family
and a single-large-transfer family rather than escalating, and later adversaries are no stronger
against the original detector than the first.

### 4.4 Phase 4 — visualisation

The Streamlit dashboard (8 pages) shows each round's decoded transaction flows scored by D_r and
D_(r+1), the evolved genomes, and — in place of an impossible 74-dimensional "decision boundary" —
two honest substitutes: exact TreeSHAP attribution drift per feature per detector, and a fixed PCA
plane of real transactions coloured by each detector's decisions. The attribution view shows the
retrained detectors relying more on amount size (`amount_log` 0.85 → 1.70) and less on account age
and threshold distance, which the adversary had learned to exploit.

### 4.5 Phase 5 — real IBM data

**Transfer between worlds.**

| recall @ budget | scored on synthetic | scored on IBM |
|---|---:|---:|
| trained on synthetic | 0.537 | 0.000 |
| trained on IBM | 0.016 | 0.668 |

Chance at this budget is about 0.005. Neither detector recognises the other world's laundering.

**What a graph-free (PaySim-style) check would keep.** Retrained on the 17 per-transaction features
alone, recall falls to 0.155 (synthetic) and 0.187 (IBM); relational features alone keep 0.455 and
0.617. PaySim, named in the synopsis as a secondary set, has no comparable account graph, so it
could only have validated the part of the detector that matters least; it was not used.

### 4.6 Phase 5 — the adversary and the arms race on IBM data

**Phase 2 on IBM (corrected decoder).** The GA reaches fitness 1.000 — but so does the hand-written
naive strategy (0.998). Detector recall on the naive strategy is **0.000**, on random genomes 0.006,
on the evolved champion 0.000. The Phase 2 gate (which expects the naive strategy to be caught) **fails**
on IBM, and is recorded as failed rather than moved: there is nothing for evolution to improve on.

**Why: one dataset shortcut.** 88% of IBM laundering is paid by ACH against 12% of benign traffic, and
the IBM detector uses it (36% of its gain). Re-scoring the naive strategy's transfers with only the
payment format changed to ACH lifts the median detector score from 0.0006 to 0.62. Retraining the
detector without the two format features (`scripts/phase5_shortcut.py`, 60 held-out episodes):

| strategy | D_0 recall / adversary wins | D_0 without format |
|---|---:|---:|
| naive | 0.000 / 100% | 0.167 / 100% |
| naive, paid by ACH | 0.144 / 100% | 0.167 / 100% |
| Phase 2 champion | 0.000 / 100% | **0.233 / 78%** |
| clean IBM holdout recall (PR-AUC) | 0.668 (0.424) | 0.636 (0.351) |

Removing the shortcut costs 3 points of recall on IBM's own laundering and makes the detector
measurably harder to evade, but it is still evaded in most episodes.

**The closed loop on IBM** (5 rounds, 30 held-out episodes per cell, same settings as the synthetic run):

| round | clean-holdout recall / PR-AUC | D_r recall vs A_r | adversary success |
|---:|---:|---:|---:|
| 0 | 0.668 / 0.424 | 0.000 | 100% |
| 1 | 0.660 / 0.420 | 0.200 | 80% |
| 2 | 0.664 / 0.390 | 0.067 | 93% |
| 3 | 0.657 / 0.407 | 0.125 | 100% |
| 4 | 0.668 / 0.399 | 0.033 | 97% |
| final D_5 | 0.669 / 0.382 | — | — |

The loop's guards held (holdout unchanged, alert budget exact every round, feedback within its 20% cap;
the Phase 3 gate passes). But **feedback did not help here.** Each retrained detector learns the
strategy it was shown (cross-play: D_1 recall 0.60 against A_0) and then loses to the next, which finds
another way through; clean-holdout PR-AUC falls from 0.424 to 0.382. On synthetic data the same loop
raised recall against the contemporary adversary from 0.125 to 0.510; on IBM it stays between 0.00 and
0.20. With only 30 episodes per cell and one seed these are indicative, not tight, but the direction is
not in doubt. A detector that leans on a dataset shortcut cannot be repaired by patching individual
strategies.


---

## 5. What the arena found in the experiment itself

The project's most reusable lesson: **a suspiciously good score is a bug until the decoded
transactions say otherwise.** Every near-perfect result was an artefact.

| when | symptom | cause | structural fix |
|---|---|---|---|
| Phase 1 | detector PR-AUC 1.000 | three generator leaks: an FX-loss band unique to laundering, laundering amounts above the benign range, laundering timestamps uniform across the clock | benign and laundering rows routed through shared functions (`fx_legs`, `_hour_of_day`), so the leak cannot recur |
| Phase 2 | detector recall 0.000 | twelve transfers at one identical timestamp, invisible to as-of features that exclude ties | per-account minimum gap in the decoder |
| Phase 3, run 1 | recall 0.000 in round 1 | one $250,000.00 transfer: a fixed round goal, while synthetic laundering is never round | per-episode goals drawn log-uniformly to the cent |
| Phase 3, run 2 | retrained detector suspiciously good | every adversary row had FX loss exactly 0.002 (a fixed fee) | fee 0; currency legs from the generator's own process |
| Phase 3, run 3 | GA fitness 0.999, held-out far lower | winner's curse: a one-transaction strategy scored 5-for-5 by luck | 200-episode re-evaluation with bootstrap intervals |
| Phase 5, IBM run 1 | evasion 1.000, completion **525×** | chained currency conversions created money; an account switching currencies every minute; a payment format that exists in IBM only as a self-transfer | on real data an operation stays in one currency and uses only accounts and formats the ledger itself uses |

The same principle fixed every one: close a leak by making both paths share code (or, on real data,
by restricting the adversary to what the data itself does), never by correcting the output.

---

## 6. Limitations

1. **No real bank data.** IBM HI-Small is the best public option but is itself simulated. The
   detector's strong typology recall partly reflects regularities of IBM's simulator (88% of its
   laundering is ACH; laundering pays new counterparties).
2. **The synthetic world has a known artefact:** its laundering is never a round amount while 21.6%
   of benign amounts are, worth about 3 points of Phase 1 recall (measured by ablation). It is absent
   in the IBM data (round amounts are 0.013% of benign and 0.022% of laundering).
3. **The adversary reads the detector's exact scores and threshold**, which a real launderer cannot.
   It is stronger than a realistic attacker, so the detector's adversarial numbers are pessimistic.
4. **One seed per closed-loop run.** Confidence intervals are over episodes, not over repeated runs.
5. **The decision boundary cannot be drawn** for a 74-dimensional model; Phase 4 shows attribution
   drift and a fixed projection instead. "Live" transaction flows are replays.
6. **PaySim was not used** (no comparable account graph or typology labels); the per-transaction
   ablation answers the question it would have answered.

---

## 7. Conclusion

The arena does what the synopsis set out to do: it measures a detector against an opponent that
adapts, round by round, and shows the result as curves rather than a single figure. Three findings
stand out.

1. **Static scores overstate robustness.** On synthetic data an evolved strategy cut recall from
   0.750 to 0.231 while delivering all of the money. On IBM data a detector with recall 0.668 caught none of the transfers of a simple strategy paid by cheque.
2. **Feedback helps only when the weakness is the strategy.** On synthetic data retraining raised recall
   against the adversary from 0.125 to 0.510 in five rounds at little cost on clean data, though the
   adversary kept succeeding in most episodes. On IBM data, where the weakness is a dataset shortcut,
   five rounds did not help (recall against the contemporary adversary 0.00-0.20, adversary success
   80-100%) and clean PR-AUC fell from 0.424 to 0.382. Removing the shortcut helped more than feedback did.
3. **An adversary is an auditor.** Its near-perfect scores exposed eight flaws in the data, the
   simulation and the experiment that ordinary evaluation had passed.

### Future work

- A shortcut-regularised detector trained with the feedback loop (the format-blind detector was only
  evaluated, not run through the arms race).
- Repeated closed-loop runs over seeds, for intervals over runs rather than episodes.
- An adversary limited to the detector's decisions (flag / no flag) rather than its scores.
- The larger IBM variants (HI-Medium, LI-Small) and graph neural network detectors.

---

## Appendix A — Reproducing every number

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"       # Python 3.12 environment
uv run python scripts/make_synthetic.py                    # synthetic world (2M rows, config/default.yaml)
uv run python scripts/train_detector.py                    # Phase 1
uv run python scripts/evolve_adversary.py                  # Phase 2
uv run python scripts/run_arena.py                         # Phase 3 (~30 min)
uv run python scripts/reevaluate_arena.py --episodes 200
uv run python scripts/build_explainability.py              # Phase 4
# Phase 5: put HI-Small_Trans.csv and HI-Small_Patterns.txt in data/raw/, then
export ARENA_CONFIG=config/ibm.yaml
uv run python scripts/prepare_ibm.py
uv run python scripts/train_detector.py
uv run python scripts/evolve_adversary.py
uv run python scripts/run_arena.py
unset ARENA_CONFIG
uv run python scripts/phase5_transfer.py
uv run pytest -q                                           # test suite
uv run streamlit run dashboard/app.py                      # dashboard
```

Every figure and table above is generated from `reports/*.json` and `reports/ibm/*.json`.
Per-phase write-ups with full detail: `docs/phase1.md` … `docs/phase5.md`.

## Appendix B — Tools

Python 3.12, LightGBM 4.7, scikit-learn, DEAP 1.4, NumPy, pandas, Streamlit, Plotly, matplotlib,
pytest, uv, Git/GitHub — matching the synopsis' software requirements.
