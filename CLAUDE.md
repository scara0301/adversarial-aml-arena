# Adversarial Fraud & AML Detection Arena

A closed-loop system that pits a transaction fraud/AML detector against a self-improving
genetic-algorithm adversary in a continuous arms race, instead of validating detection against a
single static dataset. Robustness is measured round by round against an adapting opponent.

Academic mini-project, Batch A5. Guide: Dr. B R Vatsala, Associate Professor.
Team: Anirudh Krishna (4NI24CS022), Golla Venkata Aditya (4NI24CS067), Gourav R (4NI24CS068),
Akshay MP (4NI24CS010).

## The four invariants

These are not style preferences. Each one is the difference between a real result and a
meaningless one, and each is enforced by a test. Do not weaken them for convenience.

1. **The adversary emits transactions, never features.** An adversary that edits feature vectors
   directly is cheating and voids the entire premise. Exactly one `featurize()` code path serves
   both offline training and online GA evaluation. Guarded by the batch-vs-overlay equivalence
   test in `tests/test_features.py`.

2. **Money conservation is a decoder invariant, not a repaired constraint.** The genome encodes
   only *relative* quantities (split counts, weights, fractions) and never absolute amounts. The
   decoder always allocates exactly `amount_goal`. Therefore *any* genome -- including arbitrary
   crossover offspring -- conserves money by construction. This is precisely where naive designs
   break. Guarded by `tests/test_adversary.py`.

3. **The decision threshold comes from a fixed policy, never re-tuned per round.** It is the score
   quantile that spends a fixed alert budget (default 0.5% of transactions) on the frozen clean
   holdout. If the threshold moves freely between rounds, the arms-race curve measures threshold
   drift rather than learning, and the headline result is an artefact.

4. **The frozen clean holdout never sees adversary data.** Carved out in Phase 1, written to
   `data/processed/holdout_clean.parquet`. All cross-round metrics are computed on it.
   Guarded by `tests/test_detector.py::test_holdout_contains_no_adversary_rows`.

## Two rules learned the hard way

Both cost real debugging time in Phases 1 and 2. Read them before changing the generator or the
adversary.

**A suspiciously good score is a bug, not a result.** Phase 1's first detector scored PR-AUC
1.0000 and Phase 2's first adversary achieved a detector recall of 0.0000. Both were artefacts.
Three generator leaks (an FX-loss band unique to laundering, laundering amounts above the benign
distribution, laundering timestamps uniform across the clock) and one simulation exploit (twelve
transfers at one identical timestamp, invisible to the as-of burst features because
`searchsorted` excludes ties) were responsible. When an adversarial search reports a perfect
score, assume it found a hole in the simulation and go read the decoded transactions.

**Close a leak by making both paths share code, not by correcting the output.** Every generator
leak was fixed by routing benign and laundering rows through one shared function -- `_fx_legs`,
`_hour_of_day` -- so the artefact becomes structurally impossible rather than merely corrected.
Same reasoning as having exactly one `featurize()`, and as making conservation a decoder
invariant rather than a repair step. Prefer it to patching values after the fact.

**A fixed constant in the experiment is an artefact waiting to be found.** Phase 3's first run
reached detector recall 0.000 in round 1. Reading the decoded transactions showed one $250,000.00
transfer: with a single tranche the decoder sends exactly the episode goal, every episode had the
same round goal, and the generator's laundering is *never* round while 21.6% of benign traffic is.
The detector had learned "round means benign" (`amount_mod_100` was the #2 feature all along), and
the GA found the shortcut in one round. Episode goals are now drawn per episode, log-uniformly, to
the cent. The generator artefact itself is **still there** (see `docs/phase1.md`, "A fourth
artefact"; it inflates Phase 1 by ~3 recall points) and is the first thing to fix before Phase 5.
The same run showed the closed loop works as an artefact *finder*: the next detector, trained on
that strategy, closed the hole in one round.

**On real data, the adversary may only do what the data does.** Phase 5's first IBM GA run reported
evasion 1.000 and a completion of 525x the goal. On synthetic data the decoder shares the generator's
own processes, so it cannot produce anything the world could not; real data has no generator to share.
The decoder had drawn currencies per tranche (IBM amounts are in each row's own currency, so chained
USD->Yen legs *created money*; 98% of IBM accounts use one currency) and used "Reinvestment" between two
accounts (in IBM it is only ever a self-transfer, so the detector had never seen it). `RealWorld` in
`decoder.py` now restricts a real-data operation to one currency and to the accounts and payment formats
the ledger itself uses. When adding a new data source, list what its world allows before running the GA.

## Guards on the feedback loop (Phase 3)

Retraining a detector on its adversary's output invites collapse, forgetting and overfitting.
Each guard below answers one of those and is pinned by a test in `tests/test_loop.py`. Do not
relax one without reading `src/arena/loop/pool.py` first.

- **Cap**: adversary rows are at most `arena.injection_cap` (20%) of the *original* positive count.
- **Always retrain from the original training set plus the pool**, never from the previous round's
  already-augmented set -- augmenting an augmented set compounds drift.
- **Equal row slices per round**, so an old evasion strategy is never traded away for a new one.
  Counted in rows, not operations: operations range from 2 to 100+ rows, and counting operations
  let one round starve another (a bug found and fixed in development).
- **Time windows**: training-period adversary rows come only from operations that start *and
  finish* inside the training period; validation and evaluation likewise use their own periods.
  An operation that overruns its window is dropped whole, never cut.
- **Only strategies with fitness > 0 are fed back**: fitness 0 means the goal was never completed,
  and a failed attempt is noise, not a discovered evasion.
- **Fixed episodes and a fixed decode seed for every (detector, adversary) comparison**, so two
  cells of the cross-play matrix differ because the *players* differ, never because of the dice.
- The Phase 3 gate checks that the loop and its guards work. It never requires the detector to
  win: who is ahead at round R is the finding, and a gate demanding a winner would be rigged.

## Phase status

| Phase | Objective | Scope | Status |
|-------|-----------|-------|--------|
| 0 | -- | Scaffold, canonical schema, config | done |
| 1 | 1 | Synthetic generator, loader, feature pipeline, LightGBM detector, alert-budget metrics | done — recall@budget 0.537, PR-AUC 0.454 |
| 2 | 2, 3 | GA adversary: genome, decoder, constraints, fitness, DEAP driver | done — detector recall 0.750 naive vs 0.231 evolved (re-run after the decoder fix) |
| 3 | 4 | Closed-loop arena: round orchestrator, retraining, arms-race curve | done — detector recall vs its contemporary adversary 0.125 → 0.510 over 5 rounds (adversary still wins 56% of episodes); holdout PR-AUC 0.454 → 0.421. `docs/phase3.md` |
| 4 | 5 | Streamlit + Plotly dashboard; flows, attribution drift, decision projection | done — `docs/phase4.md`; dashboard page "Phase 4 — Inside the Arms Race" |
| 5 | 1, 4 | Real IBM AML data (HI-Small) through the unchanged pipeline; transfer + ablation | done — IBM recall@budget 0.668, PR-AUC 0.424; nothing transfers between worlds. `docs/phase5.md` |
| 6 | -- | Final consolidated report | done — `docs/report.md` |

## Commands

Everything runs through `uv`. The venv is pinned to Python 3.12 (`.python-version`) -- the system
interpreter is 3.14, where LightGBM and especially DEAP wheels are a risk.

```bash
uv run python scripts/make_synthetic.py --rows 500000 --out data/synthetic/ledger.parquet
uv run python scripts/train_detector.py --ledger data/synthetic/ledger.parquet
uv run python scripts/evolve_adversary.py --detector artifacts/detector_r0.lgb
uv run python scripts/run_arena.py          # Phase 3, ~half an hour at full size
uv run python scripts/reevaluate_arena.py --episodes 200   # sharper measurement, no GA re-run
uv run python scripts/build_explainability.py               # Phase 4 views from a finished Phase 3 run
# Phase 5: real IBM data (HI-Small files in data/raw/). The SAME scripts, pointed elsewhere by an overlay:
ARENA_CONFIG=config/ibm.yaml uv run python scripts/prepare_ibm.py   # then train_detector / evolve_adversary / run_arena
uv run python scripts/phase5_transfer.py                    # synthetic <-> IBM transfer, feature-group ablation
uv run pytest -q
uv run streamlit run dashboard/app.py
```

## Module map

```
src/arena/
  schema.py              canonical ledger schema + validate(); the contract everything speaks
  data/
    synthetic.py         background traffic + injected typologies; also the test fixture
    loader.py            load_ledger("synthetic"|"ibm"|"paysim") -> canonical frame
    ibm_aml.py           real Kaggle CSV + *_Patterns.txt typology parser
  features/
    account_graph.py     AccountGraph: frozen snapshot + copy-on-write overlay  <-- the crux
    windows.py           rolling temporal aggregates
    pipeline.py          featurize(); the single shared path (invariant 1)
  detector/
    model.py             LightGBM wrapper, time-based split, frozen holdout carve-out
    threshold.py         fixed alert-budget threshold policy (invariant 3)
    metrics.py           recall@budget, PR-AUC, per-typology recall
  adversary/
    genome.py            12-gene spec with bounds
    decoder.py           genome -> transaction sub-ledger, money-conserving (invariant 2)
    constraints.py       hard repair into the plausibility envelope
    fitness.py           completion^alpha * evasion^beta - gamma*cost, with guards
    ga.py                DEAP driver -- ALL DEAP imports confined here
  loop/                  Phase 3: the closed loop
    pool.py              StrategyPool: caps, slices and time-windows the feedback to the detector
    evaluation.py        score_strategy / score_pool: fixed episodes + fixed decode seed, so cells compare
    orchestrator.py      run_arena(): rounds, retraining, cross-play matrix, runtime invariant checks
    reevaluate.py        re-measure a finished run on more episodes with bootstrap CIs (no GA re-run)
    gate.py              the Phase 3 gate (checks the loop works; never requires the detector to win)
    report.py            the four Phase 3 figures + summary, redrawable from reports/phase3_metrics.json
  explain.py             Phase 4: TreeSHAP attribution drift, fixed projection, decoded flows
  config.py              YAML config; ARENA_CONFIG lays an overlay (config/ibm.yaml) over the defaults
dashboard/               Streamlit explainer (see dashboard/README.md)
```

### Why `AccountGraph` looks the way it does

A GA run is population 100 x 40 generations x 5 episodes = ~20,000 candidate evaluations.
Re-featurizing a multi-million-row background ledger per candidate is impossible. So:

- `AccountGraph.build(background)` does one O(N) pass, once per run. The background never
  changes between arms-race rounds -- adversary data is fed to the detector as training rows and
  is never added to the background ledger -- so the frozen graph is reused for every round.
- `AccountGraph.overlay(sub_ledger)` returns a copy-on-write view. Only accounts the sub-ledger
  touches (tens, bounded by hops x mules) get modified counters; everything else reads through to
  the frozen snapshot.
- `featurize()` then computes features for only the sub-ledger rows against that overlay.

This is what makes Phase 2 tractable at all, and it is the reason invariant 1 is achievable
without a prohibitive cost.

## Conventions

- **Never report accuracy.** At a 0.1% positive rate it is ~99.9% for a model that predicts all
  zeros. ROC-AUC may be reported but must be labelled as misleading at this prevalence. The
  primary metric is **recall at a 0.5% alert budget**; PR-AUC is the threshold-free sanity check.
- **Time-based splits only.** A random split leaks future graph state into the past.
- **DEAP stays inside `ga.py`.** If it misbehaves, a ~150-line hand-rolled loop over a 12-gene
  vector replaces it without touching genome, decoder, constraints or fitness.
- Config lives in `config/default.yaml`; scripts take CLI overrides.
- Each phase writes metrics JSON + figures to `reports/`, a writeup to `docs/phaseN.md`, and a
  notebook to `notebooks/`.
- Scripts print a **phase gate** (PASS/FAIL) against explicit thresholds. If a gate has to move,
  say so in the writeup with the evidence — `docs/phase1.md` records why the PR-AUC bar went from
  0.60 to 0.40 after a capacity sweep showed 0.50 was this data's ceiling.
- When writing files from a shell one-liner on Windows, pass `encoding="utf-8"` explicitly.
  Python's default `cp1252` cannot encode an en dash or arrow, and the partial write truncates
  the target file — this destroyed this file once already.
- `data/` and `artifacts/` are gitignored and regenerable. `reports/` is committed -- it is the
  deliverable.

## Honest limitations to keep stating

Recorded so they are not rediscovered at demo time:

1. **You cannot plot the decision boundary** of a 74-dimensional LightGBM model, despite the
   synopsis phrasing. The substitutes are a 2-D PCA projection with score contours and -- more
   informative -- SHAP feature-importance drift between rounds, which shows *which* features the
   adversary forced the detector to stop trusting.
2. **"Live transaction flows" is replay**, not a live feed. Fine for a demo; describe it honestly.
3. **PaySim does not transfer cleanly.** Different schema, no comparable account graph or typology
   labels, so the relational features do not carry over. Fallback: validate only the
   per-transaction feature subset.
4. **Synthetic numbers are a pipeline check.** Detectors trained on synthetic data and on IBM catch
   essentially none of each other's laundering (Phase 5: 0.000 and 0.016 recall). And the IBM set is
   itself simulated (AMLworld): 88% of its laundering is ACH, a shortcut the IBM detector learned.
5. **The generator's laundering is never a round number**, while 21.6% of benign amounts are. That
   is a free tell the detector learned (`amount_mod_100`, the #2 feature), worth about +3 points of
   recall and +2 of PR-AUC in Phase 1 (measured by ablation, `docs/phase1.md`). Real laundering is
   often round. Phase 3 closes the adversary's access to it; the artefact is unfixed.
6. **Phase 3's adversary can read the detector's exact scores** and threshold, which a real
   launderer cannot. That makes it stronger than a realistic attacker, so the detector's numbers
   are conservative on that axis.
