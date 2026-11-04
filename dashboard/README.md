# Dashboard

An interactive Streamlit explainer for Phases 1-3.

```bash
uv run streamlit run dashboard/app.py
```

## What it is, and isn't

Six of the seven pages are read-only: they load `reports/phase{1,2,3}_metrics.json` and the
`reports/*.png` figures those scripts already wrote, and never touch a trained model. The seventh,
**Try the Adversary Yourself**, is the exception — it loads the real round-0 detector
(`artifacts/detector_r0.lgb`) and the frozen background graph, and lets a visitor set the
adversary's twelve genes with sliders and score the result live.

The Phase 3 page shows the arms-race curve with its bootstrap intervals, the cross-play matrix, and
the clean-holdout cost, all computed from the JSON with no hard-coded numbers. It is a replay of a
finished run, not a live feed, and it is not the final Phase 4 dashboard (no SHAP drift view yet).

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
