"""Streamlit dashboard: an interactive walkthrough of the arena prototype.

Explains what Phases 0-2 built without retraining or re-evolving anything --
every static number and chart on the first two results pages comes straight
from reports/phase{1,2}_metrics.json, the artifacts scripts/train_detector.py
and scripts/evolve_adversary.py already wrote. The one exception is "Try the
Adversary Yourself", which loads the actual trained detector and lets you
decode and score a genome live, through the exact featurize() path that
produced every number on the static pages -- see dashboard/live.py.

Run with:
    uv run streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# dashboard/ is not an installed package (unlike src/arena, via `pip install
# -e .`), so its parent -- the repo root -- has to be on sys.path before the
# `from dashboard import ...` lines below will resolve, regardless of exactly
# how the `streamlit run` invocation happens to set up sys.path.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from dashboard import data  # noqa: E402
from dashboard.views import (  # noqa: E402
    how_it_works,
    limitations,
    overview,
    phase1_detector,
    phase2_adversary,
    phase3_arms_race,
    phase4_inside,
    phase5_ibm,
    try_it,
)

st.set_page_config(page_title="Adversarial AML Arena", page_icon="🕵️", layout="wide")

PAGES = {
    "🏠 Overview": overview.render,
    "🛡️ Phase 1 — The Detector": phase1_detector.render,
    "🧬 Phase 2 — The Adversary": phase2_adversary.render,
    "🥊 Phase 3 — The Arms Race": phase3_arms_race.render,
    "🔬 Phase 4 — Inside the Arms Race": phase4_inside.render,
    "🌍 Phase 5 — Real IBM Data": phase5_ibm.render,
    "🧪 Try the Adversary Yourself": try_it.render,
    "⚙️ How It Works": how_it_works.render,
    "⚠️ Limitations & What's Next": limitations.render,
}


def main() -> None:
    st.sidebar.title("🕵️ Adversarial AML Arena")
    st.sidebar.caption("Batch A5 mini-project")
    choice = st.sidebar.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
    st.sidebar.divider()
    _render_sidebar_status()
    PAGES[choice]()


def _render_sidebar_status() -> None:
    p1_ok = data.phase1_metrics() is not None
    p2_ok = data.phase2_metrics() is not None
    p3_ok = data.phase3_metrics() is not None
    st.sidebar.markdown("**Phase status**")
    st.sidebar.markdown(f"- Phase 1 (detector): {'✅' if p1_ok else '⬜ not run'}")
    st.sidebar.markdown(f"- Phase 2 (adversary): {'✅' if p2_ok else '⬜ not run'}")
    st.sidebar.markdown(f"- Phase 3 (closed loop): {'✅' if p3_ok else '⬜ not run'}")
    p4_ok = data.phase4_explain() is not None
    st.sidebar.markdown(f"- Phase 4 (visualisation): {'✅' if p4_ok else '⬜ not run'}")
    p5_ok = data.ibm("phase1_metrics") is not None
    st.sidebar.markdown(f"- Phase 5 (real IBM data): {'✅' if p5_ok else '⬜ not run'}")
    if not (p1_ok and p2_ok):
        st.sidebar.warning(
            "Run the Phase 1/2 scripts to populate this dashboard with real "
            "numbers:\n\n```\nuv run python scripts/make_synthetic.py\n"
            "uv run python scripts/train_detector.py\n"
            "uv run python scripts/evolve_adversary.py\n```"
        )


if __name__ == "__main__":
    main()
