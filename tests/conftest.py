"""Shared fixtures.

Everything is built from the synthetic generator rather than from a checked-in
data file, so the tests exercise the same code path the real pipeline uses and
cannot drift out of sync with the schema.

Sizes are small: these fixtures run for every test, so they trade statistical
realism for speed. Tests that need a well-fit detector build their own.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from arena.adversary.decoder import Episode
from arena.data.synthetic import generate
from arena.features import AccountGraph


@pytest.fixture(scope="session")
def ledger() -> pd.DataFrame:
    """A small synthetic ledger with all five typologies present."""
    return generate(accounts=800, rows=12_000, laundering_rate=0.03, days=45, seed=7)


@pytest.fixture(scope="session")
def graph(ledger: pd.DataFrame) -> AccountGraph:
    return AccountGraph.build(ledger)


@pytest.fixture(scope="session")
def episode(ledger: pd.DataFrame) -> Episode:
    """A laundering task using real accounts from the fixture ledger."""
    accounts = sorted(
        set(ledger["from_account"].astype(str)) | set(ledger["to_account"].astype(str))
    )
    return Episode(
        src_account=accounts[0],
        dst_account=accounts[1],
        amount_goal=250_000.0,
        deadline_hours=336.0,
        start_hour=240.0,
        mule_pool=accounts[2:16],
        bank_of={},
        epoch=pd.Timestamp(ledger["timestamp"].min()),
    )


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)
