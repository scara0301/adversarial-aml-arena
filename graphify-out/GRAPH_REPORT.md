# Graph Report - .  (2026-09-17)

## Corpus Check
- Corpus is ~35,335 words - fits in a single context window. You may not need a graph.

## Summary
- 582 nodes · 1108 edges · 18 communities detected
- Extraction: 58% EXTRACTED · 41% INFERRED · 1% AMBIGUOUS · INFERRED: 453 edges (avg confidence: 0.71)
- Token cost: 36,800 input · 27,400 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Design Rationale & Invariants|Design Rationale & Invariants]]
- [[_COMMUNITY_Adversary Genome & Decoder|Adversary Genome & Decoder]]
- [[_COMMUNITY_GA Engine & Episode Sampling|GA Engine & Episode Sampling]]
- [[_COMMUNITY_Detector Training & Config|Detector Training & Config]]
- [[_COMMUNITY_Canonical Schema & Data Loading|Canonical Schema & Data Loading]]
- [[_COMMUNITY_Feature Pipeline & Account Graph|Feature Pipeline & Account Graph]]
- [[_COMMUNITY_Metrics & Threshold Policy|Metrics & Threshold Policy]]
- [[_COMMUNITY_Metric Policy & Project Conventions|Metric Policy & Project Conventions]]
- [[_COMMUNITY_Synthetic Ledger Generator|Synthetic Ledger Generator]]
- [[_COMMUNITY_As-Of Window Queries|As-Of Window Queries]]
- [[_COMMUNITY_Limitations & Data Provenance|Limitations & Data Provenance]]
- [[_COMMUNITY_GA Convergence Figure|GA Convergence Figure]]
- [[_COMMUNITY_Feature Importance Figure|Feature Importance Figure]]
- [[_COMMUNITY_PR Curve Figure|PR Curve Figure]]
- [[_COMMUNITY_Recall Comparison Figure|Recall Comparison Figure]]
- [[_COMMUNITY_Typology Recall Figure|Typology Recall Figure]]
- [[_COMMUNITY_Orphan Graph Build Docstring|Orphan: Graph Build Docstring]]
- [[_COMMUNITY_Orphan Package Init|Orphan: Package Init]]

## God Nodes (most connected - your core abstractions)
1. `Detector` - 41 edges
2. `Episode` - 34 edges
3. `AccountGraph` - 32 edges
4. `main()` - 24 edges
5. `decode()` - 23 edges
6. `featurize()` - 22 edges
7. `main()` - 21 edges
8. `FitnessConfig` - 21 edges
9. `Outcome` - 18 edges
10. `ImplausibleStrategy` - 17 edges

## Surprising Connections (you probably didn't know these)
- `Known Limitations Stated Up Front` --semantically_similar_to--> `Limitation: PaySim Does Not Transfer Cleanly`  [INFERRED] [semantically similar]
  README.md → CLAUDE.md
- `LightGBM Detector Over 74 Features` --semantically_similar_to--> `74 Features Across Three Families`  [INFERRED] [semantically similar]
  README.md → docs/phase1.md
- `Detector, threshold policy and metrics tests.  The threshold tests cover invaria` --uses--> `Detector`  [INFERRED]
  tests\test_detector.py → src\arena\detector\model.py
- `Invariant 3: the budget determines the operating point.      If the realised fla` --uses--> `Detector`  [INFERRED]
  tests\test_detector.py → src\arena\detector\model.py
- `A collapsed model produces mostly identical scores; this must not crash.` --uses--> `Detector`  [INFERRED]
  tests\test_detector.py → src\arena\detector\model.py

## Hyperedges (group relationships)
- **The Four Invariants Enforced by Tests** — claude_invariant_1_transactions_not_features, claude_invariant_2_money_conservation_decoder, claude_invariant_3_fixed_threshold_policy, claude_invariant_4_frozen_clean_holdout, readme_test_suite_64_tests [EXTRACTED 1.00]
- **The Five Laundering Typologies** — phase1_typology_fan_out, phase1_typology_fan_in, phase1_typology_cycle, phase1_typology_gather_scatter, phase1_typology_stack, phase2_gene_typology_template [EXTRACTED 1.00]
- **The Degenerate-Optimum Guards in the Fitness Function** — phase2_guard_move_zero_amount, phase2_guard_one_giant_transfer, phase2_guard_dust_shredding, phase2_guard_deadline_stall, phase2_guard_episode_averaging, phase2_guard_fitness_sharing_clones, phase2_fitness_formula [EXTRACTED 1.00]

## Communities

### Community 0 - "Design Rationale & Invariants"
Cohesion: 0.03
Nodes (97): AccountGraph: Frozen Snapshot Plus Copy-on-Write Overlay, Why AccountGraph Looks the Way It Does (20,000 Evaluations), Single Shared featurize() Code Path, Invariant 1: Adversary Emits Transactions, Never Features, Invariant 2: Money Conservation Is a Decoder Invariant, Limitation: PaySim Does Not Transfer Cleanly, Rule: Close a Leak by Making Both Paths Share Code, Rule: A Suspiciously Good Score Is a Bug, Not a Result (+89 more)

### Community 1 - "Adversary Genome & Decoder"
Cohesion: 0.06
Nodes (72): AssertionError, conservation_error(), ImplausibleStrategy, repair_genome(), validate_sub_ledger(), decode(), delivered_amount(), elapsed_hours() (+64 more)

### Community 2 - "GA Engine & Episode Sampling"
Cohesion: 0.06
Nodes (47): episode(), graph(), ledger(), Shared fixtures.  Everything is built from the synthetic generator rather than f, A small synthetic ledger with all five typologies present., A laundering task using real accounts from the fixture ledger., The plausibility envelope, enforced by hard repair.  Objective 2 requires the, Absolute money-conservation error, in currency units.      Should be ~0 for ev (+39 more)

### Community 3 - "Detector Training & Config"
Cohesion: 0.07
Nodes (37): Config, load(), Config loading. One YAML file, dotted-path access, CLI-overridable., Nested dict with dotted-path lookup.      Thin on purpose: the value of a config, ``cfg.get("detector.lightgbm.num_leaves")``., A config path resolved against the repo root., parse_patterns(), Parse ``*_Patterns.txt`` into a join key table.      Returns a frame of ``[times (+29 more)

### Community 4 - "Canonical Schema & Data Loading"
Cohesion: 0.06
Nodes (41): load_ibm_aml(), Reader for the real IBM AML dataset (Kaggle: ealtman2019).  Two files per varian, Load the IBM AML transactions CSV into the canonical schema.      Args:, load_ledger(), Single entry point for every data source.  Everything downstream calls :func:`lo, Load a ledger in the canonical schema.      Args:         source: ``"synthetic"`, Validate and write a ledger to parquet, creating parent directories., save_ledger() (+33 more)

### Community 5 - "Feature Pipeline & Account Graph"
Cohesion: 0.07
Nodes (37): AccountGraph, build(), _build_state(), CycleIndex, Account graph: frozen snapshot plus copy-on-write overlay.  This module exists t, Assemble one account's state, including the single ordered walk.      The walk i, Directed adjacency supporting bounded cycle queries.      Answers "is this trans, A copy with extra edges, for the overlay path.          Only the touched account (+29 more)

### Community 6 - "Metrics & Threshold Policy"
Cohesion: 0.07
Nodes (37): evaluate(), precision_at_threshold(), Metrics for a ~0.1% positive rate.  At this prevalence most familiar metrics are, Fraction of true positives scoring at or above ``tau``., Fraction of alerts that are true positives -- the analyst's hit rate., Recall at ``tau``, broken out per laundering typology.      A transaction is cou, Full metric bundle for one evaluation set.      Returns a JSON-serialisable dict, recall_at_threshold() (+29 more)

### Community 7 - "Metric Policy & Project Conventions"
Cohesion: 0.11
Nodes (26): Adversarial Fraud & AML Detection Arena, Batch A5 Team and Internal Guide, config/default.yaml With CLI Overrides, Convention: DEAP Stays Inside ga.py, Invariant 3: Threshold From Fixed Alert-Budget Policy, Invariant 4: Frozen Clean Holdout Never Sees Adversary Data, Convention: Never Report Accuracy, Phase Status Table (Phases 0-6) (+18 more)

### Community 8 - "Synthetic Ledger Generator"
Cohesion: 0.12
Nodes (23): account_ids(), _benign_amounts(), _business_hour_offsets(), _fx_legs(), _hour_of_day(), _inject_patterns(), _launder_rows(), _make_background() (+15 more)

### Community 9 - "As-Of Window Queries"
Cohesion: 0.17
Nodes (19): AccountState, Time-sorted event history for one account, with prefix sums.      Split by direc, As-of account features for one side of the transaction.      Rows are grouped by, _side_features(), _as_of_index(), counts_and_volume(), pair_repetition(), As-of window queries against an :class:`~arena.features.account_graph.AccountSta (+11 more)

### Community 10 - "Limitations & Data Provenance"
Cohesion: 0.12
Nodes (17): Canonical Ledger Schema: The Contract Everything Speaks, Limitation: 'Live Transaction Flows' Is Replay, Limitation: Cannot Plot a High-Dimensional Decision Boundary, Limitation: Synthetic Typologies Are Cleaner Than Real Ones, Convention: Scripts Print a PASS/FAIL Phase Gate, reports/ Is Committed; data/ and artifacts/ Are Regenerable, SHAP Feature-Importance Drift Between Rounds, Limitation: Synthetic Data Is Not Real Data (+9 more)

### Community 11 - "GA Convergence Figure"
Cohesion: 0.18
Nodes (17): Y Axis: component value (0.0 to 1.0), Y Axis: fitness (0.0 to ~0.9), X Axis: generation (0 to 40), Behaviour: Best-Fitness Plateau near 0.90 after Generation 8, Behaviour: Completion Saturated at 1.0 Throughout, Bottom Panel: Fitness Components vs Generation, Behaviour: Population Diversity Holds Roughly Flat near 0.35-0.40, Insight: Elitism Locks the Champion While Selection Still Lifts the Population (+9 more)

### Community 12 - "Feature Importance Figure"
Cohesion: 0.21
Nodes (15): Account-age and recency features (rcv_age_h, snd_age_h, snd_since_prev_h), Account-graph relational feature family (snd_out_deg, snd_in_uniq_cp, snd_pair_reps, rcv_out_vol, snd_in_vol, snd_out_vol), Insight: the top-ranked amount and threshold features are the most attractive levers for the GA adversary to manipulate when evading detection, Insight: amount features dominate; amount_log gain is ~3x the runner-up and ~9x rank 5, so the detector leans heavily on amount structure over graph topology, Amount-based feature family (amount_log, amount_mod_100, threshold_gap, amount_vs_snd_mean, snd_mean_out_amt), amount_log (rank 1, gain ~350,000 — dominant feature), amount_mod_100 (rank 2, gain ~110,000 — round-number structuring signal), day_of_week (rank 10, gain ~18,000 — calendar timing feature) (+7 more)

### Community 13 - "PR Curve Figure"
Cohesion: 0.25
Nodes (14): Legend Annotation: 0.5% alert budget, Phase 1 AML Fraud Detector, Chance Baseline (dashed grey, prevalence=0.2264%), Blue PR Curve (monotone decreasing, precision ~1.0 until recall 0.15, near 0 by recall 1.0), Figure: Detector Precision-Recall Curve (Phase 1), Data Source: Frozen Clean Holdout, Insight: detector precision is roughly two orders of magnitude above chance at the operating point, Marked Operating Point: recall=0.537, precision=0.243 (+6 more)

### Community 14 - "Recall Comparison Figure"
Cohesion: 0.33
Nodes (12): Ambiguous: alert budget / threshold policy behind these recall values is not printed on the figure, Transaction fraud/AML detector under evaluation in this comparison, Implication: random genomes evade no better than hand-written (0.755 vs 0.767); GA search, not randomness, drives the evasion, Evolved (GA best) laundering strategy - detector recall 0.247, Figure: Detector recall - naive vs evolved laundering (Phase 2 headline bar chart), Genetic-algorithm adversary (GA best individual produced the evolved strategy), Metric axis: detector recall on the adversary's transactions (0.0 to 1.0), Naive (hand-written) laundering strategy - detector recall 0.767 (+4 more)

### Community 15 - "Typology Recall Figure"
Cohesion: 0.33
Nodes (11): Typology: cycle - recall 0.481, Phase 1 detector evaluated in this figure (LightGBM under the fixed alert-budget policy), Ambiguity: evaluation split is not labelled on the figure (frozen clean holdout assumed, not printed), Typology: fan-in - recall 0.411 (worst detected), Typology: fan-out - recall 0.414, Figure: Recall by Laundering Typology (Phase 1 horizontal bar chart), Typology: gather-scatter - recall 0.458, Metric: recall at the fixed alert budget (x-axis, 0.0 to 1.0) (+3 more)

### Community 16 - "Orphan: Graph Build Docstring"
Cohesion: 1.0
Nodes (1): One O(N) pass over ``ledger``. Run once per arms-race round.

### Community 17 - "Orphan: Package Init"
Cohesion: 1.0
Nodes (0): 

## Ambiguous Edges - Review These
- `Adversarial Fraud & AML Detection Arena` → `Convention: Explicit UTF-8 Encoding on Windows Writes`  [AMBIGUOUS]
  CLAUDE.md · relation: conceptually_related_to
- `Scale: 2M Rows, 20k Accounts, 0.2% Laundering` → `Guard: Fitness Averaged Over 5 Independent Episodes`  [AMBIGUOUS]
  docs/phase2.md · relation: conceptually_related_to
- `Phase 1 Figure: Top 20 Features by Gain (horizontal bar chart)` → `Account-age and recency features (rcv_age_h, snd_age_h, snd_since_prev_h)`  [AMBIGUOUS]
  reports/phase1_feature_importance.png · relation: shares_data_with
- `Figure: Detector Precision-Recall Curve (Phase 1)` → `PR-AUC Value (not printed on the figure; must be read from the metrics JSON)`  [AMBIGUOUS]
  reports/phase1_pr_curve.png · relation: references
- `Blue PR Curve (monotone decreasing, precision ~1.0 until recall 0.15, near 0 by recall 1.0)` → `PR-AUC Value (not printed on the figure; must be read from the metrics JSON)`  [AMBIGUOUS]
  reports/phase1_pr_curve.png · relation: conceptually_related_to
- `Figure: Recall by Laundering Typology (Phase 1 horizontal bar chart)` → `Ambiguity: evaluation split is not labelled on the figure (frozen clean holdout assumed, not printed)`  [AMBIGUOUS]
  reports/phase1_typology_recall.png · relation: conceptually_related_to
- `Y Axis: component value (0.0 to 1.0)` → `Behaviour: Population Diversity Holds Roughly Flat near 0.35-0.40`  [AMBIGUOUS]
  reports/phase2_convergence.png · relation: conceptually_related_to
- `Series: population mean fitness (dashed grey)` → `Series: population diversity (dotted purple)`  [AMBIGUOUS]
  reports/phase2_convergence.png · relation: semantically_similar_to
- `Figure: Detector recall - naive vs evolved laundering (Phase 2 headline bar chart)` → `Ambiguous: alert budget / threshold policy behind these recall values is not printed on the figure`  [AMBIGUOUS]
  reports/phase2_recall_comparison.png · relation: conceptually_related_to
- `Metric axis: detector recall on the adversary's transactions (0.0 to 1.0)` → `Ambiguous: alert budget / threshold policy behind these recall values is not printed on the figure`  [AMBIGUOUS]
  reports/phase2_recall_comparison.png · relation: conceptually_related_to

## Knowledge Gaps
- **108 isolated node(s):** `Config loading. One YAML file, dotted-path access, CLI-overridable.`, `Nested dict with dotted-path lookup.      Thin on purpose: the value of a config`, ```cfg.get("detector.lightgbm.num_leaves")``.`, `A config path resolved against the repo root.`, `Canonical ledger schema.  Every data source -- the synthetic generator, the real` (+103 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Orphan: Graph Build Docstring`** (1 nodes): `One O(N) pass over ``ledger``. Run once per arms-race round.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Orphan: Package Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Adversarial Fraud & AML Detection Arena` and `Convention: Explicit UTF-8 Encoding on Windows Writes`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Scale: 2M Rows, 20k Accounts, 0.2% Laundering` and `Guard: Fitness Averaged Over 5 Independent Episodes`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Phase 1 Figure: Top 20 Features by Gain (horizontal bar chart)` and `Account-age and recency features (rcv_age_h, snd_age_h, snd_since_prev_h)`?**
  _Edge tagged AMBIGUOUS (relation: shares_data_with) - confidence is low._
- **What is the exact relationship between `Figure: Detector Precision-Recall Curve (Phase 1)` and `PR-AUC Value (not printed on the figure; must be read from the metrics JSON)`?**
  _Edge tagged AMBIGUOUS (relation: references) - confidence is low._
- **What is the exact relationship between `Blue PR Curve (monotone decreasing, precision ~1.0 until recall 0.15, near 0 by recall 1.0)` and `PR-AUC Value (not printed on the figure; must be read from the metrics JSON)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Figure: Recall by Laundering Typology (Phase 1 horizontal bar chart)` and `Ambiguity: evaluation split is not labelled on the figure (frozen clean holdout assumed, not printed)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Y Axis: component value (0.0 to 1.0)` and `Behaviour: Population Diversity Holds Roughly Flat near 0.35-0.40`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._