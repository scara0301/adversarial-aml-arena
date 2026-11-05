# Dashboard

An interactive Streamlit dashboard for Phases 1-5 (objective 5).

```bash
uv run streamlit run dashboard/app.py
```

## What it is, and isn't

Eight of the nine pages are read-only: they load `reports/phase{1,2,3}_metrics.json`,
`reports/phase4_explain.json`, `reports/ibm/*.json` and the figures those scripts already wrote, and
never touch a trained model. The ninth,
**Try the Adversary Yourself**, is the exception — it loads the real round-0 detector
(`artifacts/detector_r0.lgb`) and the frozen background graph, and lets a visitor set the
adversary's twelve genes with sliders and score the result live.

The Phase 3 page shows the arms-race curve with its bootstrap intervals, the cross-play matrix, and
the clean-holdout cost, all computed from the JSON with no hard-coded numbers. It is a replay of a
finished run, not a live feed.

The Phase 4 page is the round explorer: pick an adversary and one of its decoded operations, and see
the transfers drawn left to right and scored by D_r (the detector it beat) or D_(r+1) (the one
retrained on it); then TreeSHAP attribution drift per feature per detector, and a fixed projection
of real transactions coloured by each detector's decisions. It replaces a "decision boundary" plot,
which cannot be drawn honestly for a 74-dimensional model (see `docs/phase4.md`).

The Phase 5 page sets the IBM results beside the synthetic ones: cleaning, detector metrics and
typology recall, the cross-dataset transfer matrix, feature-group ablations, the adversary, and the
IBM arms race.

## Layout

```
dashboard/
  app.py            entry point — sidebar nav, run with `streamlit run dashboard/app.py`
  data.py           cached loaders for reports/*.json and reports/*.csv
  live.py           the one place that loads the real detector + AccountGraph (st.cache_resource)
  views/
    overview.py         pitch, headline result, the four invariants, phase status
    phase1_detector.py  metrics, PR curve, typology recall, feature importance
    phase2_adversary.py convergence, recall gap, winning strategy decoded
    phase3_arms_race.py  arms-race curve + CIs, cross-play matrix, holdout cost, what was fed back
    phase4_inside.py     flow explorer, attribution drift heatmap, fixed decision projection
    phase5_ibm.py        the real IBM data beside the synthetic results
    try_it.py            live genome explorer (sliders -> decode -> score)
    how_it_works.py      architecture walkthrough: the shared featurize(), the overlay,
                          money conservation, the simulation exploit the GA found
    limitations.py       honest caveats, carried over from docs/phase{1,2,3}.md
```

Named `views/`, not `pages/`, on purpose: Streamlit auto-builds its own sidebar navigation from
any folder literally named `pages` next to the running script, which would collide with this
app's own sidebar radio in `app.py`.

## Data freshness

`data.py`'s loaders are cached by file path, not by file content or mtime. If you regenerate
`reports/` while the dashboard is running (e.g. re-run `train_detector.py`), restart Streamlit,
or press "r" in the running app, to pick up the new numbers.

## If `artifacts/` or `data/` don't exist

Both are gitignored and regenerable, so a fresh clone won't have them. The static pages show a
warning with the commands to run; the live explorer checks `live.artifacts_available()` first and
shows the same warning instead of crashing.
