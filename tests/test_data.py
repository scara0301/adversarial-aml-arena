"""Schema and generator tests.

The generator tests are not decoration. Two separate leaks were found in it
during Phase 1 -- an FX-loss band unique to laundering, and laundering amounts
sitting entirely above the benign distribution -- each of which produced a
detector that scored beautifully and had learned nothing transferable. The
distribution-overlap tests below exist so that neither can come back unnoticed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from arena import schema
from arena.data.synthetic import generate
from arena.schema import REPORTING_THRESHOLD, SchemaError


def test_generated_ledger_validates(ledger):
    schema.validate(ledger)
    assert list(ledger.columns) == list(schema.COLUMNS)


def test_all_five_typologies_are_present(ledger):
    present = set(ledger["typology"].dropna().unique())
    assert present == set(schema.TYPOLOGIES), f"missing typologies: {set(schema.TYPOLOGIES) - present}"


def test_laundering_rate_is_close_to_target():
    df = generate(accounts=500, rows=20_000, laundering_rate=0.01, days=30, seed=1)
    assert 0.008 <= df["is_laundering"].mean() <= 0.013


def test_labels_and_typologies_are_consistent(ledger):
    assert (ledger.loc[ledger["typology"].notna(), "is_laundering"] == 1).all()
    assert ledger.loc[ledger["is_laundering"] == 1, "typology"].notna().all()
    assert ledger.loc[ledger["is_laundering"] == 1, "pattern_id"].notna().all()


def test_ledger_is_time_sorted(ledger):
    assert ledger["timestamp"].is_monotonic_increasing


def test_no_self_transfers(ledger):
    assert not (ledger["from_account"] == ledger["to_account"]).any()


def test_fx_loss_distributions_overlap(ledger):
    """Regression guard: the first leak found in Phase 1.

    Laundering rows used to always keep the same currency and lose 0-0.5% in
    transit, while benign rows lost either exactly 0% or 0.5-3%. Laundering
    therefore occupied an FX-loss band benign traffic never touched, and that one
    feature reached PR-AUC 1.00 -- six times the gain of everything else combined.

    Both classes now draw FX behaviour from the same function, so the *fraction
    of cross-currency transfers* must be similar between them.
    """
    cross = ledger["payment_currency"].astype(str) != ledger["receiving_currency"].astype(str)
    benign_rate = cross[ledger["is_laundering"] == 0].mean()
    launder_rate = cross[ledger["is_laundering"] == 1].mean()
    assert abs(benign_rate - launder_rate) < 0.06, (
        f"cross-currency rate differs too much (benign {benign_rate:.3f} vs "
        f"laundering {launder_rate:.3f}) -- the FX leak may have returned"
    )


def test_amount_distributions_overlap(ledger):
    """Regression guard: the second leak found in Phase 1.

    Laundering tranches used to sit entirely above the benign amount
    distribution, so ``amount_log`` alone accounted for 97% of the model's gain.
    Structuring exists precisely to hide inside normal amounts, so the
    interquartile ranges must genuinely overlap.
    """
    benign = ledger.loc[ledger["is_laundering"] == 0, "amount_paid"]
    launder = ledger.loc[ledger["is_laundering"] == 1, "amount_paid"]

    b_lo, b_hi = np.percentile(benign, [25, 75])
    l_lo, l_hi = np.percentile(launder, [25, 75])
    overlap = max(0.0, min(b_hi, l_hi) - max(b_lo, l_lo))
    assert overlap > 0, (
        f"benign IQR [{b_lo:.0f}, {b_hi:.0f}] and laundering IQR "
        f"[{l_lo:.0f}, {l_hi:.0f}] do not overlap -- amount alone separates the classes"
    )


def test_benign_traffic_exists_near_the_threshold(ledger):
    """Some legitimate transfers must sit just under $10,000.

    If none did, "just below the threshold" would be a free detection signal and
    the adversary's structuring problem would be trivial.
    """
    benign = ledger.loc[ledger["is_laundering"] == 0, "amount_paid"]
    near = ((benign >= 0.85 * REPORTING_THRESHOLD) & (benign < REPORTING_THRESHOLD)).sum()
    assert near > 0, "no benign traffic near the reporting threshold"


def test_laundering_has_diurnal_structure(ledger):
    """Laundering must not be uniformly spread across the clock.

    A flat hour distribution for laundering against business-hours-clustered
    benign traffic was a third easy separator; real launderers operate inside
    normal hours.
    """
    hours = pd.to_datetime(ledger.loc[ledger["is_laundering"] == 1, "timestamp"]).dt.hour
    in_business = ((hours >= 9) & (hours < 17)).mean()
    # Uniform would be 8/24 = 0.33; benign is far higher. Laundering should sit
    # clearly above uniform without matching benign exactly.
    assert in_business > 0.40, f"laundering looks uniform across the day ({in_business:.3f})"


def test_generation_is_reproducible():
    a = generate(accounts=200, rows=3_000, laundering_rate=0.02, days=20, seed=42)
    b = generate(accounts=200, rows=3_000, laundering_rate=0.02, days=20, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_give_different_ledgers():
    a = generate(accounts=200, rows=3_000, laundering_rate=0.02, days=20, seed=1)
    b = generate(accounts=200, rows=3_000, laundering_rate=0.02, days=20, seed=2)
    assert not a["amount_paid"].equals(b["amount_paid"])


# ---------------------------------------------------------------- schema itself


def test_validate_rejects_negative_amounts(ledger):
    bad = ledger.head(10).copy()
    bad.loc[bad.index[0], "amount_paid"] = -5.0
    with pytest.raises(SchemaError, match="strictly positive"):
        schema.validate(bad)


def test_validate_rejects_self_transfers(ledger):
    bad = ledger.head(10).copy()
    bad.loc[bad.index[0], "to_account"] = bad.loc[bad.index[0], "from_account"]
    with pytest.raises(SchemaError, match="self-transfer"):
        schema.validate(bad)


def test_validate_rejects_inconsistent_labels(ledger):
    bad = ledger.head(10).copy()
    bad.loc[bad.index[0], "typology"] = "fan-out"
    bad.loc[bad.index[0], "is_laundering"] = 0
    with pytest.raises(SchemaError, match="typology must have"):
        schema.validate(bad)


def test_validate_rejects_unknown_typology(ledger):
    bad = ledger.head(10).copy()
    bad.loc[bad.index[0], "typology"] = "smurfing-deluxe"
    bad.loc[bad.index[0], "is_laundering"] = 1
    with pytest.raises(SchemaError, match="unknown typology"):
        schema.validate(bad)


def test_coerce_reports_missing_columns(ledger):
    with pytest.raises(SchemaError, match="missing required column"):
        schema.coerce(ledger.drop(columns=["payment_format"]))


def test_empty_ledger_has_the_full_schema():
    empty = schema.empty_ledger()
    assert list(empty.columns) == list(schema.COLUMNS)
    schema.validate(empty)
