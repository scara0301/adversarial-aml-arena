"""Phase 5: the real-data path, tested on small fixtures in the IBM file format."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from arena.adversary.decoder import RealWorld, decode, delivered_amount, real_world
from arena.adversary.fitness import EpisodeSampler
from arena.adversary.ga import naive_genome
from arena.adversary.genome import random_genome
from arena.config import Config
from arena.data.ibm_aml import load_ibm_aml, parse_patterns

HEADER = ("Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,"
          "Amount Paid,Payment Currency,Payment Format,Is Laundering\n")


def _row(ts, fb, fa, tb, ta, amt, cur="US Dollar", fmt="Cheque", y=0, recv=None, rcur=None):
    return f"{ts},{fb:03d},{fa},{tb:03d},{ta},{recv or amt:.2f},{rcur or cur},{amt:.2f},{cur},{fmt},{y}\n"


@pytest.fixture
def ibm_files(tmp_path):
    rows = []
    # ten live days of benign traffic, 200 rows a day, in three currencies -- one of
    # them (Yen) outside the synthetic vocabulary, as fifteen of IBM's are
    # them (Yen) outside the synthetic vocabulary. As in IBM, currency belongs to
    # the account: A{i} and B{j} transact only with accounts of their own currency.
    curs = ("US Dollar", "Euro", "Yen")
    for d in range(1, 11):
        for k in range(200):
            i = k % 40
            j = (i + 3 * (k // 40 + 1)) % 42   # same residue mod 3, so same currency
            rows.append(_row(f"2022/09/{d:02d} {k % 24:02d}:{k % 60:02d}", 1, f"A{i}", 2,
                             f"B{j}", 100.0 + k, cur=curs[i % 3]))
    # a self-transfer, one of them laundering
    rows.append(_row("2022/09/03 10:00", 1, "A1", 1, "A1", 500.0, fmt="Reinvestment"))
    rows.append(_row("2022/09/03 11:00", 1, "A2", 1, "A2", 600.0, y=1))
    # a fan-out pattern, listed in the patterns file
    fan = [_row(f"2022/09/05 0{h}:10", 3, "L0", 4, f"M{h}", 2000.0 + h, fmt="ACH", y=1) for h in range(4)]
    rows += fan
    # a positive the patterns file does not mention
    rows.append(_row("2022/09/06 12:00", 3, "L9", 4, "M9", 4321.0, fmt="ACH", y=1))
    # the dead tail: two days of almost nothing, mostly laundering
    rows.append(_row("2022/09/12 01:00", 3, "L5", 4, "M5", 900.0, y=1))
    rows.append(_row("2022/09/13 01:00", 1, "A3", 2, "B3", 800.0))

    trans = tmp_path / "HI-Small_Trans.csv"
    trans.write_text(HEADER + "".join(rows), encoding="utf-8")
    pats = tmp_path / "HI-Small_Patterns.txt"
    pats.write_text("BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 4-degree Fan-Out\n" + "".join(fan)
                    + "END LAUNDERING ATTEMPT - FAN-OUT\n\n", encoding="utf-8")
    return trans, pats


def test_patterns_file_is_parsed(ibm_files):
    _, pats = ibm_files
    p = parse_patterns(pats)
    assert len(p) == 4 and set(p["typology"]) == {"fan-out"} and p["pattern_id"].nunique() == 1


def test_self_transfers_are_dropped_and_counted(ibm_files):
    df = load_ibm_aml(ibm_files[0])
    c = df.attrs["cleaning"]
    assert (df["from_account"] != df["to_account"]).all()
    assert c["self_transfers_dropped"] == 2 and c["self_transfer_positives_dropped"] == 1


def test_the_dead_tail_is_cut(ibm_files):
    df = load_ibm_aml(ibm_files[0])
    c = df.attrs["cleaning"]
    assert df["timestamp"].max() < pd.Timestamp("2022-09-11")
    assert c["tail_rows_dropped"] == 2 and c["tail_positives_dropped"] == 1
    kept = load_ibm_aml(ibm_files[0], trim_tail=False)
    assert kept["timestamp"].max() >= pd.Timestamp("2022-09-13")


def test_unmatched_positives_are_unattributed_not_stack(ibm_files):
    df = load_ibm_aml(ibm_files[0])
    pos = df[df["is_laundering"] == 1]
    assert (pos["typology"] == "fan-out").sum() == 4
    assert pos.loc[pos["from_account"] == "L9", "typology"].tolist() == ["unattributed"]
    assert "stack" not in set(pos["typology"])
    assert df.attrs["cleaning"]["positives_matched_to_pattern"] == 4


def test_synthetic_ledgers_keep_the_synthetic_processes(ledger):
    assert real_world(ledger) is None


def test_a_real_ledger_gets_its_own_rules(ibm_files):
    df = load_ibm_aml(ibm_files[0])
    w = real_world(df)
    assert isinstance(w, RealWorld) and w.currency == "US Dollar"
    # Reinvestment only ever appeared as a (dropped) self-transfer here
    assert "Reinvestment" not in w.formats and "Cheque" in w.formats
    usd = set(df.loc[df["payment_currency"] == "US Dollar", "from_account"].astype(str))
    assert w.accounts and w.accounts <= usd | set(df["to_account"].astype(str))
    pay, recv, amt = w.legs(np.random.default_rng(0), np.array([10.0, 20.0]))
    assert set(pay) == set(recv) == {"US Dollar"}
    np.testing.assert_array_equal(amt, [10.0, 20.0])


def test_episodes_on_real_data_use_only_accounts_in_the_currency(ibm_files):
    df = load_ibm_aml(ibm_files[0])
    s = EpisodeSampler(df, df["timestamp"].min())
    assert s.world is not None and set(s.accounts) <= s.world.accounts
    for ep in s.sample(20, np.random.default_rng(1), 20_000.0, 48.0):
        assert {ep.src_account, ep.dst_account, *ep.mule_pool} <= s.world.accounts


def test_money_is_conserved_on_the_real_data_path(ibm_files):
    """The regression test for the first IBM run, whose champion delivered 525x its goal
    by chaining currency conversions. Any genome, on real-data rules, conserves money."""
    df = load_ibm_aml(ibm_files[0])
    s = EpisodeSampler(df, df["timestamp"].min())
    rng = np.random.default_rng(7)
    n_checked = 0
    for k in range(300):
        ep = s.sample(1, rng, 25_000.0, 48.0)[0]
        sub = decode(random_genome(rng), ep, np.random.default_rng(k), 0.0)
        if sub.empty:
            continue
        n_checked += 1
        assert delivered_amount(sub, ep) <= ep.amount_goal + 1e-6
        assert set(sub["payment_currency"].astype(str)) == {"US Dollar"}
        assert (sub["amount_received"] == sub["amount_paid"]).all()
        assert set(sub["payment_format"].astype(str)) <= set(s.world.formats)
    assert n_checked > 100


def test_config_overlay_replaces_only_its_keys(monkeypatch, tmp_path):
    base = Config.load()
    over = tmp_path / "o.yaml"
    over.write_text("paths:\n  reports: elsewhere\nadversary:\n  episode:\n    deadline_hours: 24\n",
                    encoding="utf-8")
    monkeypatch.setenv("ARENA_CONFIG", str(over))
    cfg = Config.load()
    assert cfg.get("paths.reports") == "elsewhere"
    assert cfg.get("adversary.episode.deadline_hours") == 24
    assert cfg.get("adversary.episode.amount_goal") == base.get("adversary.episode.amount_goal")
    assert cfg.get("paths.detector") == base.get("paths.detector")
    assert cfg.get("detector.lightgbm") == base.get("detector.lightgbm")
