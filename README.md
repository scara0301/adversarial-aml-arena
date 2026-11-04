# Adversarial Fraud & AML Detection Arena

A closed-loop system that pits a transaction fraud/AML detector against a self-improving
genetic-algorithm adversary in a continuous arms race, instead of validating detection against a
single static dataset. Detector robustness is measured round by round against an opponent that
adapts, rather than reported as one accuracy figure against fraud patterns already known at
training time.

Academic mini-project, Batch A5.
Internal guide: **Dr. B R Vatsala**, Associate Professor.

| USN | Name |
|---|---|
| 4NI24CS022 | Anirudh Krishna |
| 4NI24CS067 | Golla Venkata Aditya |
| 4NI24CS068 | Gourav R |
| 4NI24CS010 | Akshay MP |

## Quick start

Requires Python 3.12 (pinned in `.python-version`) and [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"
```

```bash
uv run python scripts/make_synthetic.py
```

```bash
uv run python scripts/train_detector.py
```

```bash
uv run python scripts/evolve_adversary.py
```

```bash
uv run python scripts/run_arena.py
```

```bash
uv run python scripts/reevaluate_arena.py --episodes 200
```

```bash
uv run pytest -q
```

The first script writes a 2M-row synthetic ledger; the second trains the detector and freezes the
clean holdout; the third evolves the adversary against it; the fourth runs the closed loop for five
rounds of retraining (about half an hour); the fifth re-measures that run on more episodes with
confidence intervals, without re-running any search. Each writes metrics and figures to `reports/`.

## Dashboard

An interactive Streamlit explainer, built ahead of the final Phase 4 dashboard, that walks through
what Phases 1-3 produced. Run it after the commands above:

```bash
uv run streamlit run dashboard/app.py
```

Seven pages: an overview with the headline result and the four invariants, a Phase 1 page (detector
metrics, PR curve, typology recall, feature importance), a Phase 2 page (GA convergence, the
naive-vs-evolved recall gap, the winning strategy decoded), a Phase 3 page (the arms-race curve with
confidence intervals, the cross-play matrix, the clean-holdout cost), a live genome explorer that decodes
and scores any set of the twelve genes against the actual trained detector, an architecture
walkthrough, and a limitations page. Everything except the live explorer reads the static JSON in
`reports/` rather than retraining or re-evolving anything — see `dashboard/README.md` for how it's
put together.

This is not the final Phase 4 dashboard: everything is a replay of a finished run, not a live feed,
and the SHAP drift view is not built.

## Phase status

| Phase | Objective | Scope | Status |
|---|---|---|---|
| 0 | — | Scaffold, canonical schema, config | done |
| 1 | 1 | Synthetic generator, loader, feature pipeline, LightGBM detector, alert-budget metrics | done |
| 2 | 2, 3 | GA adversary: genome, decoder, constraints, fitness, DEAP driver | done |
| 3 | 4 | Closed-loop arena: round orchestrator, retraining, arms-race curve | done |
| 4 | 5 | Streamlit + Plotly dashboard | prototype explainer built ahead of Phase 3 (`dashboard/`) |
| 5 | — | Real IBM AML data, PaySim cross-validation | not started |
| 6 | — | Final consolidated report | not started |

Writeups with results and design reasoning: [docs/phase1.md](docs/phase1.md),
[docs/phase2.md](docs/phase2.md), [docs/phase3.md](docs/phase3.md). Notebooks:
`notebooks/phase1_detector.ipynb`, `notebooks/phase2_adversary.ipynb`,
`notebooks/phase3_arms_race.ipynb`.

## How it works

**The detector** is a LightGBM classifier over 74 features in three families: per-transaction
(amount, currency, format, time-of-day, and proximity to the $10,000 reporting threshold),
account aggregates (degree, unique counterparties, volume, flow ratio, reciprocity,
inter-arrival), and relational (rolling 1h/24h/7d bursts, a gather-scatter score, and closed-cycle
flags for lengths 2–4). Every account feature is computed *as of* the transaction's own timestamp.

**The adversary** is a genetic algorithm over a 12-gene genome encoding structuring (tranche
count, skew, threshold proximity), timing (delays, jitter, business-hours alignment, hop
dormancy) and routing (typology template, hop depth, mule pool size, mule reuse). A decoder turns
a genome into actual transactions, which are then featurized and scored by the same pipeline that
produced the detector's training data.

**The interface between them** is the crux, and it is a single design decision: the adversary
emits *transactions*, never feature vectors. An `AccountGraph` frozen snapshot with a
copy-on-write `overlay()` makes that affordable — 4.8 ms per candidate against a 2M-row
background, versus the impossible cost of re-featurizing the whole ledger 20,000 times.

## The four invariants

These are enforced by tests, not conventions. Each is the difference between a real result and a
meaningless one.

1. **The adversary emits transactions, never features.** Exactly one `featurize()` serves both
   training and adversary evaluation. An adversary editing features directly would be cheating.
   → `tests/test_features.py::test_batch_and_overlay_agree`
2. **Money conservation is a decoder invariant, not a repaired constraint.** No gene holds an
   absolute amount, so any crossover offspring still moves exactly the goal amount.
   → `tests/test_adversary.py::test_money_is_conserved_under_crossover`
3. **The decision threshold comes from a fixed alert budget, never re-tuned per round.**
   Otherwise the arms-race curve measures threshold drift rather than learning.
   → `tests/test_detector.py::test_threshold_spends_exactly_the_alert_budget`
4. **The frozen clean holdout never sees adversary data.** All cross-round metrics come from it.
   → `tests/test_detector.py::test_holdout_contains_no_adversary_rows`

## Data

Development runs on a synthetic generator that emits the exact IBM AML schema with injected,
labelled typologies, so the pipeline runs with no multi-hundred-megabyte download. The real
Kaggle dataset ("IBM Transactions for Anti Money Laundering", `ealtman2019`) drops in behind the
same loader interface:

```bash
uv run python -c "from arena.data import load_ledger; print(load_ledger('ibm', path='data/raw/HI-Small_Trans.csv', nrows=100000).head())"
```

`src/arena/data/ibm_aml.py` reads the transactions CSV *and* parses the companion
`*_Patterns.txt`, which is the only source of typology labels — the CSV alone gives just a binary
flag.

## Known limitations

Stated up front rather than discovered at demo time:

1. **You cannot plot the decision boundary** of a 74-dimensional LightGBM model, despite the
   synopsis phrasing. The substitutes are a 2-D projection with score contours and — more
   informative — SHAP feature-importance drift between rounds, which shows *which* features the
   adversary forced the detector to stop trusting.
2. **"Live transaction flows" is replay**, not a live feed.
3. **PaySim does not transfer cleanly.** Different schema, no comparable account graph or typology
   labels, so the relational features do not carry over. Fallback: validate only the
   per-transaction subset.
4. **Synthetic typologies are cleaner than real ones.** Phase 1 and 2 numbers are a pipeline
   check, not the project's headline result.

## Repository layout

```
src/arena/
  schema.py              canonical ledger schema; the contract everything speaks
  config.py              YAML config with dotted-path access
  data/                  synthetic generator, loader, real IBM AML reader
  features/              AccountGraph (frozen + overlay), windows, the shared featurize
  detector/              LightGBM model, alert-budget threshold, imbalance-aware metrics
  adversary/             genome, decoder, constraints, fitness, DEAP driver
scripts/                 make_synthetic, train_detector, evolve_adversary
notebooks/               one per phase
tests/                   64 tests, including the four invariant checks
config/default.yaml      all tunables
reports/                 metrics JSON + figures (committed; this is the deliverable)
docs/                    per-phase writeups
```

`data/` and `artifacts/` are gitignored and fully regenerable from the scripts above.

## Stack

Python 3.12 · LightGBM · scikit-learn · DEAP · NumPy · Pandas · Matplotlib · Plotly · Streamlit ·
SHAP · pytest

Python is pinned to 3.12 rather than the system's 3.14: LightGBM and especially DEAP are
C-extension packages whose 3.14 wheel availability is uncertain, and nothing here needs 3.14. All
DEAP usage is confined to `src/arena/adversary/ga.py`, so it can be replaced by a hand-rolled
loop without touching the genome, decoder, constraints or fitness.
