"""Phase 4: the explainability views must be faithful to the models they explain."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from arena.adversary.ga import naive_genome
from arena.detector import time_split, train
from arena.explain import attribution_drift, decoded_flows, fixed_projection, flow_layout, mean_abs_shap
from arena.features import featurize


@pytest.fixture(scope="module")
def detectors(ledger, graph):
    split = time_split(ledger, 0.6, 0.15)
    d_a, _ = train(split, graph, params={"n_estimators": 30}, negative_ratio=10.0)
    d_b, _ = train(split, graph, params={"n_estimators": 30, "seed": 3, "num_leaves": 7}, negative_ratio=10.0)
    return [d_a, d_b], split


@pytest.fixture(scope="module")
def X_holdout(detectors, graph):
    (d, _), split = detectors
    return featurize(split.holdout, graph, d.window_hours)


def test_shap_contributions_sum_to_the_raw_score(detectors, X_holdout):
    """The identity that makes the attributions exact rather than approximate."""
    d = detectors[0][0]
    X = X_holdout.reindex(columns=d.features, fill_value=0.0).to_numpy(dtype=np.float64)[:300]
    contrib = d.booster.predict(X, pred_contrib=True)
    raw = d.booster.predict(X, raw_score=True)
    assert contrib.shape == (len(X), len(d.features) + 1)
    np.testing.assert_allclose(contrib.sum(axis=1), raw, atol=1e-6)


def test_mean_abs_shap_is_nonnegative_and_indexed_by_feature(detectors, X_holdout):
    d = detectors[0][0]
    s = mean_abs_shap(d, X_holdout)
    assert list(s.index) == list(d.features)
    assert (s >= 0).all() and s.sum() > 0


def test_attribution_drift_shapes_and_drift_column(detectors, X_holdout):
    dets = detectors[0]
    out = attribution_drift(dets, {"a": X_holdout, "b": X_holdout.iloc[:200]}, top=5)
    assert out["detectors"] == ["D_0", "D_1"]
    assert len(out["features"]) == 5
    for name in ("a", "b"):
        M = np.array(out["sets"][name]["mean_abs_shap"])
        assert M.shape == (5, 2)
        np.testing.assert_allclose(out["sets"][name]["drift"], M[:, 1] - M[:, 0])
    assert out["sets"]["b"]["n_rows"] == 200


def test_projection_is_fixed_and_deterministic(X_holdout):
    fit_on = X_holdout.iloc[:500]
    groups = {"all": X_holdout, "head": X_holdout.iloc[:50]}
    c1, ev1 = fixed_projection(fit_on, groups, seed=0)
    c2, ev2 = fixed_projection(fit_on, groups, seed=0)
    assert c1["all"].shape == (len(X_holdout), 2)
    np.testing.assert_allclose(c1["all"], c2["all"])
    # a point's position does not depend on which group it is projected with
    np.testing.assert_allclose(c1["head"], c1["all"][:50])
    assert 0 < sum(ev1) <= 1 and ev1 == ev2


def test_projection_is_not_dominated_by_one_heavy_tailed_feature():
    """Signed log1p: a single 1e9 outlier must not define the plane."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(400, 4)), columns=list("abcd"))
    X.loc[0, "a"] = 1e9
    coords, _ = fixed_projection(X, {"x": X})
    spread = np.abs(coords["x"]).max(axis=0) / np.abs(coords["x"][1:]).max(axis=0)
    assert (spread < 10).all()


def test_flow_layout_puts_source_first_and_destination_last():
    edges = pd.DataFrame({
        "from_account": ["S", "S", "M1", "M2", "M3"],
        "to_account": ["M1", "M2", "M3", "D", "D"],
    })
    pos = flow_layout(edges, "S", "D")
    assert set(pos) == {"S", "M1", "M2", "M3", "D"}
    assert pos["S"][0] == 0.0
    assert pos["D"][0] == 1.0
    assert all(0 < p[0] < 1 for a, p in pos.items() if a.startswith("M"))
    assert pos["M1"][0] < pos["M3"][0]
    assert all(0 < p[1] < 1 for p in pos.values())


def test_flow_layout_of_a_direct_transfer():
    pos = flow_layout(pd.DataFrame({"from_account": ["S"], "to_account": ["D"]}), "S", "D")
    assert pos == {"S": (0.0, 0.5), "D": (1.0, 0.5)}


def test_decoded_flows_are_scored_by_both_judges(detectors, graph, episode):
    dets = detectors[0]
    taus = [0.5, 0.5]
    flows = decoded_flows([naive_genome()], [episode], dets, taus, graph, 0.0, seed=1)
    assert len(flows) == 1 and flows[0]["judges"] == ["D_0", "D_1"]
    ep = flows[0]["episodes"][0]
    n = len(ep["transfers"]["from"])
    assert n > 0
    for j in ("D_0", "D_1"):
        assert len(ep["scores"][j]["score"]) == n
        assert ep["scores"][j]["flagged"] == [s >= 0.5 for s in ep["scores"][j]["score"]]
    assert set(ep["layout"]) >= set(ep["transfers"]["from"]) | set(ep["transfers"]["to"])
    assert ep["layout"][ep["src"]][0] == 0.0 and ep["layout"][ep["dst"]][0] == 1.0


def test_the_last_round_is_judged_by_its_own_detector_only(detectors, graph, episode):
    dets = detectors[0]
    flows = decoded_flows([naive_genome(), naive_genome()], [episode], dets, [0.5, 0.5], graph, 0.0, seed=1)
    assert flows[1]["judges"] == ["D_1"]
