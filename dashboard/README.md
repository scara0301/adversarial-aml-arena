# Dashboard

An interactive Streamlit explainer for Phases 1-2, built ahead of Phase 3.

```bash
uv run streamlit run dashboard/app.py
```

## What it is, and isn't

Five of the six pages are read-only: they load `reports/phase{1,2}_metrics.json` and the
`reports/*.png` figures those two scripts already wrote, and never touch the trained model.
The sixth page, **Try the Adversary Yourself**, is the exception — it loads the real
`artifacts/detector_r0.lgb` and the frozen background graph, and lets a visitor set the
adversary's twelve genes with sliders and score the result live.

This is **not** the full Phase 4 arena dashboard. There is no round-by-round arms-race curve
here, because that data doesn't exist until Phase 3's closed-loop orchestrator is built — this
shows round 0 only: one trained detector, one evolved adversary, no retraining loop between
them.

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
    try_it.py            live genome explorer (sliders -> decode -> score)
    how_it_works.py      architecture walkthrough: the shared featurize(), the overlay,
                          money conservation, the simulation exploit the GA found
    limitations.py       honest caveats, carried over from docs/phase{1,2}.md
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
