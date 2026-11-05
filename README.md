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
uv run python scripts/build_explainability.py
```

Phase 5 needs the IBM files (`HI-Small_Trans.csv`, `HI-Small_Patterns.txt`, from Kaggle) in
`data/raw/`, then the same scripts under the IBM overlay (`export ARENA_CONFIG=config/ibm.yaml`):
`prepare_ibm.py`, `train_detector.py`, `evolve_adversary.py`, `run_arena.py`; and, without the overlay,
`scripts/phase5_transfer.py`. See [docs/phase5.md](docs/phase5.md).

```bash
uv run pytest -q
```

The first script writes a 2M-row synthetic ledger; the second trains the detector and freezes the
clean holdout; the third evolves the adversary against it; the fourth runs the closed loop for five
rounds of retraining (about half an hour); the fifth re-measures that run on more episodes with
confidence intervals, without re-running any search. Each writes metrics and figures to `reports/`.

## Dashboard

An interactive Streamlit dashboard that walks through what every phase produced. Run it after the
commands above:

```bash
uv run streamlit run dashboard/app.py
```

Nine pages: an overview with the headline result and the four invariants, a Phase 1 page (detector
metrics, PR curve, typology recall, feature importance), a Phase 2 page (GA convergence, the
naive-vs-evolved recall gap, the winning strategy decoded), a Phase 3 page (the arms-race curve with
confidence intervals, the cross-play matrix, the clean-holdout cost), a Phase 4 page (each round's decoded transaction flows scored by D_r and
D_(r+1), TreeSHAP attribution drift, and a fixed projection of real transactions coloured by each
detector's decisions), a Phase 5 page (the same pipeline on the real IBM data), a live genome explorer that decodes
and scores any set of the twelve genes against the actual trained detector, an architecture
walkthrough, and a limitations page. Everything except the live explorer reads the static JSON in
`reports/` rather than retraining or re-evolving anything — see `dashboard/README.md` for how it's
put together.

Everything is a replay of a finished run, not a live feed.

## Phase status

| Phase | Objective | Scope | Status |
|---|---|---|---|
| 0 | — | Scaffold, canonical schema, config | done |
| 1 | 1 | Synthetic generator, loader, feature pipeline, LightGBM detector, alert-budget metrics | done |
| 2 | 2, 3 | GA adversary: genome, decoder, constraints, fitness, DEAP driver | done |
| 3 | 4 | Closed-loop arena: round orchestrator, retraining, arms-race curve | done |
| 4 | 5 | Dashboard: flows, attribution drift, decision projection | done |
| 5 | 1, 4 | Real IBM AML data (HI-Small), cross-dataset transfer, feature-group ablation | done |
| 6 | — | Final consolidated report | done — [docs/report.md](docs/report.md) |

**Final report: [docs/report.md](docs/report.md).** Writeups with results and design reasoning:
[docs/phase1.md](docs/phase1.md), [docs/phase2.md](docs/phase2.md), [docs/phase3.md](docs/phase3.md),
[docs/phase4.md](docs/phase4.md), [docs/phase5.md](docs/phase5.md). Notebooks:
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
   synopsis phrasing. The substitutes (Phase 4) are TreeSHAP attribution drift between rounds, which
   shows *which* features the adversary forced the detector to stop trusting, and a fixed projection
   of real transactions coloured by each detector's decisions — no synthesised contours.
2. **"Live transaction flows" is replay**, not a live feed.
3. **PaySim was not used.** No comparable account graph or typology labels, so 57 of 74 features do
   not carry over. Phase 5 measures what such a check would keep instead: retrained on the 17
   per-transaction features alone, recall falls under a third on both datasets.
4. **Synthetic numbers are a pipeline check.** Detectors trained on the synthetic world and on IBM
   catch essentially none of each other's laundering (Phase 5). The IBM data is the headline, and it
   is itself simulated (IBM's AMLworld) — the best public benchmark, not bank data.

## Repository layout

```
src/arena/
  schema.py              canonical ledger schema; the contract everything speaks
  config.py              YAML config with dotted-path access
  data/                  synthetic generator, loader, real IBM AML reader
  features/              AccountGraph (frozen + overlay), windows, the shared featurize
  detector/              LightGBM model, alert-budget threshold, imbalance-aware metrics
  adversary/             genome, decoder, constraints, fitness, DEAP driver
  loop/                  Phase 3 closed loop: pool, evaluation, orchestrator, re-evaluation, gate
  explain.py             Phase 4: attribution drift, fixed projection, decoded flows
scripts/                 one per step; prepare_ibm and phase5_transfer for Phase 5
dashboard/               Streamlit dashboard
notebooks/               one per phase
tests/                   102 tests, including the four invariant checks
config/default.yaml      all tunables; config/ibm.yaml is the Phase 5 overlay (ARENA_CONFIG)
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
