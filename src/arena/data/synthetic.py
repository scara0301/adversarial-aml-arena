"""Synthetic ledger generator.

Emits the canonical schema with realistic benign background traffic plus
injected, labelled laundering typologies. Two jobs:

1. Let Phases 1-3 be developed and run with no multi-hundred-megabyte download,
   behind the same loader interface the real IBM AML data will use.
2. Serve as the fixture for every test in the project.

Realism decisions that matter for the science, not just for looks:

* **Power-law counterparty degree.** Most accounts transact with a handful of
  others; a few hubs transact with thousands. Without this, account degree is a
  near-perfect laundering giveaway and the detector's job becomes trivial.
* **Diurnal and weekly rhythm.** Benign traffic clusters in business hours.
  This is what gives the adversary's ``business_hours_bias`` gene something real
  to exploit.
* **Round-number clustering in benign amounts.** Real people send $500, not
  $487.33. If only laundering produced round numbers, ``amount % 1000`` would be
  a free win for the detector.
* **Benign traffic near the reporting threshold.** Some legitimate transactions
  sit just under $10,000. If none did, structuring would be trivially detectable
  and the whole arms race would be vacuous.

The last two are the easiest to get wrong and the most damaging: a generator
that makes laundering *too* separable produces a detector with a great score and
no ability to generalise, and an adversary with nothing to learn.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from arena import schema
from arena.schema import CURRENCIES, PAYMENT_FORMATS, REPORTING_THRESHOLD

# Benign traffic leans on low-scrutiny formats; wires and cash are rarer.
_BENIGN_FORMAT_P = np.array([0.30, 0.20, 0.22, 0.06, 0.12, 0.08, 0.02])

# Laundering traffic shifts toward formats that move value fast between banks.
# The shift is kept moderate on purpose: a sharper contrast (wire 0.32 vs 0.12,
# credit card 0.04 vs 0.22) made payment format close to a giveaway on its own.
# Launderers do prefer wires and cash, but they operate inside normal traffic,
# so the distributions have to overlap heavily.
_LAUNDER_FORMAT_P = np.array([0.28, 0.14, 0.10, 0.05, 0.24, 0.14, 0.05])

CURRENCY_P = np.array([0.55, 0.20, 0.13, 0.10, 0.02])

_TYPOLOGY_CYCLE = ("fan-out", "fan-in", "cycle", "gather-scatter", "stack")


def account_ids(n: int) -> np.ndarray:
    """Stable account identifiers. Exposed so the adversary can pick mules."""
    return np.array([f"ACC{i:07d}" for i in range(n)], dtype=object)


def _powerlaw_activity(rng: np.random.Generator, n: int) -> np.ndarray:
    """Per-account transaction propensity, heavy-tailed.

    Pareto rather than uniform so the degree distribution has real hubs. Without
    hubs, out-degree alone separates fan-out laundering perfectly and the
    detector learns nothing that would transfer to real data.
    """
    raw = rng.pareto(1.6, size=n) + 1.0
    return raw / raw.sum()


def _hour_of_day(rng: np.random.Generator, n: int, p_business: float = 0.90) -> np.ndarray:
    """Hour-of-day with morning and afternoon peaks plus an overnight tail.

    Shared by benign and laundering traffic, with only ``p_business`` differing.
    That sharing is deliberate. An earlier version gave laundering a flat
    ``uniform(0, 24)`` hour while benign traffic was business-hours clustered,
    which handed the detector a near-perfect separator that has no counterpart in
    real data. Real launderers mimic normal hours, so laundering gets a slightly
    weaker business-hours bias -- a genuine signal, but not a decisive one.
    """
    peak = rng.random(n)
    lo, hi = p_business * 0.5, p_business
    hour = np.where(
        peak < lo,
        rng.normal(10.5, 1.6, size=n),
        np.where(peak < hi, rng.normal(15.0, 2.0, size=n), rng.uniform(0, 24, size=n)),
    )
    return np.clip(hour, 0, 23.999)


def fx_legs(
    rng: np.random.Generator, amounts: np.ndarray, cur_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Receiving currency and received amount, from one shared process.

    Returns ``(payment_currency, receiving_currency, amount_received)``.

    **This function exists because of a measured leak.** Laundering rows used to
    always keep the same currency and lose 0-0.5% in transit, while benign rows
    lost either exactly 0% (same currency) or 0.5-3% (cross currency). Laundering
    therefore occupied an FX-loss band that benign traffic never touched, and
    ``fx_loss`` alone scored PR-AUC 1.0 -- six times the gain of every other
    feature combined. It was an artefact of the generator, not a property of
    money laundering.

    Routing both classes through this single function makes the leak
    structurally impossible rather than merely fixed, which is the same reason
    the feature pipeline has exactly one ``featurize``.
    """
    n = len(amounts)
    cur = np.asarray(CURRENCIES)[cur_idx]
    same = rng.random(n) < 0.93
    recv_cur = np.where(same, cur, np.asarray(CURRENCIES)[(cur_idx + 1) % len(CURRENCIES)])
    recv_amt = np.where(same, amounts, amounts * rng.uniform(0.97, 0.995, size=n))
    return cur, recv_cur, recv_amt


def _business_hour_offsets(rng: np.random.Generator, n: int, days: int) -> np.ndarray:
    """Timestamps as hours-from-start, with diurnal and weekly structure."""
    day = rng.integers(0, days, size=n).astype(np.float64)

    # Weekends are quieter: push ~60% of would-be weekend activity to weekdays.
    is_weekend = (day % 7) >= 5
    shift = rng.random(n) < 0.6
    day = np.where(is_weekend & shift, day - 2.0, day)
    day = np.clip(day, 0, days - 1)

    return day * 24.0 + _hour_of_day(rng, n, p_business=0.90)


def _benign_amounts(rng: np.random.Generator, n: int) -> np.ndarray:
    """Log-normal amounts, with round-number and near-threshold clusters.

    The near-threshold cluster is deliberate and important: it stops "just under
    $10,000" from being a free detection signal, and so makes the adversary's
    structuring problem non-trivial.

    The distribution's scale is also load-bearing. An earlier version used
    ``lognormal(6.5, 1.35)`` -- a median of about $665 -- while structured
    laundering tranches landed between $1k and $10k. Laundering was therefore
    systematically *larger* than benign traffic, and ``amount_log`` alone
    accounted for 97% of the model's gain. The detector had learned an amount
    threshold, not laundering. Structuring exists precisely to hide inside the
    normal amount range, so the two distributions have to overlap.
    """
    amt = rng.lognormal(mean=7.6, sigma=1.5, size=n)

    # ~22% of benign traffic is a round number.
    round_mask = rng.random(n) < 0.22
    step = rng.choice([100.0, 500.0, 1000.0, 5000.0], size=n, p=[0.45, 0.25, 0.22, 0.08])
    amt = np.where(round_mask, np.maximum(step, np.round(amt / step) * step), amt)

    # ~1.5% of benign traffic legitimately sits just under the threshold.
    near_mask = rng.random(n) < 0.015
    amt = np.where(near_mask, rng.uniform(0.88, 0.995, size=n) * REPORTING_THRESHOLD, amt)

    return np.clip(amt, 10.0, 5_000_000.0)


def _make_background(
    rng: np.random.Generator,
    accounts: np.ndarray,
    banks: np.ndarray,
    rows: int,
    days: int,
    start: pd.Timestamp,
) -> pd.DataFrame:
    """Benign traffic. Vectorised end to end -- this is the bulk of the rows."""
    n_acc = len(accounts)
    activity = _powerlaw_activity(rng, n_acc)

    src = rng.choice(n_acc, size=rows, p=activity)
    dst = rng.choice(n_acc, size=rows, p=activity)

    # Resolve self-transfers by nudging, rather than resampling in a loop.
    clash = src == dst
    if clash.any():
        dst[clash] = (dst[clash] + 1 + rng.integers(0, n_acc - 1, size=int(clash.sum()))) % n_acc

    hours = _business_hour_offsets(rng, rows, days)
    amounts = _benign_amounts(rng, rows)
    fmt_idx = rng.choice(len(PAYMENT_FORMATS), size=rows, p=_BENIGN_FORMAT_P)
    cur_idx = rng.choice(len(CURRENCIES), size=rows, p=CURRENCY_P)

    cur, recv_cur, recv_amt = fx_legs(rng, amounts, cur_idx)

    return pd.DataFrame(
        {
            "timestamp": start + pd.to_timedelta(hours, unit="h"),
            "from_bank": banks[src],
            "from_account": accounts[src],
            "to_bank": banks[dst],
            "to_account": accounts[dst],
            "amount_paid": amounts,
            "payment_currency": cur,
            "amount_received": recv_amt,
            "receiving_currency": recv_cur,
            "payment_format": np.asarray(PAYMENT_FORMATS)[fmt_idx],
            "is_laundering": np.zeros(rows, dtype=np.int8),
            "typology": pd.array([None] * rows, dtype="string"),
            "pattern_id": pd.array([None] * rows, dtype="Int64"),
            "source": np.full(rows, "background", dtype=object),
        }
    )


def structured_split(rng: np.random.Generator, total: float, k: int) -> np.ndarray:
    """Split ``total`` into ``k`` tranches that hug the reporting threshold.

    This is the *hand-written* structuring strategy the ground-truth typologies
    use, and it doubles as the naive baseline the evolved adversary is measured
    against in Phase 2. Deliberately competent but unsophisticated: tranches
    land just under the threshold with modest jitter, which is exactly the kind
    of regularity a gradient-boosted detector picks up.
    """
    weights = rng.dirichlet(np.full(k, 6.0))
    tranches = total * weights
    # Pull anything over the threshold down under it. The jitter range is wide
    # (0.45-0.99 rather than 0.82-0.985) because a narrow band piles every
    # tranche into $8.2k-$9.9k, which is itself a glaring regularity -- and one
    # a real launderer would avoid. Wide jitter keeps the tranches sub-threshold
    # without making them identically suspicious.
    over = tranches > REPORTING_THRESHOLD
    if over.any():
        tranches[over] = REPORTING_THRESHOLD * rng.uniform(0.45, 0.99, size=int(over.sum()))
    return np.maximum(tranches, 100.0)


def _launder_rows(
    rng: np.random.Generator,
    pattern_id: int,
    typology: str,
    hops: list[tuple[int, int, float]],
    accounts: np.ndarray,
    banks: np.ndarray,
    base_hour: float,
    start: pd.Timestamp,
) -> pd.DataFrame:
    """Turn a list of (src_idx, dst_idx, amount) hops into canonical rows."""
    n = len(hops)
    src = np.array([h[0] for h in hops])
    dst = np.array([h[1] for h in hops])
    amt = np.array([h[2] for h in hops], dtype=np.float64)

    # Legs are spread over a few days inside the pattern's window, then placed
    # within each day using the *same* diurnal process as benign traffic. See
    # _hour_of_day: giving laundering a flat hour distribution made it trivially
    # separable.
    # Whole days, not fractional ones. A fractional day offset multiplied by 24
    # adds a random hour shift that cancels out the diurnal placement below --
    # the same mistake as an unsnapped base_hour, and it left laundering
    # uniformly spread across the clock.
    day_offset = np.sort(rng.integers(0, 5, size=n)).astype(np.float64)
    offsets = day_offset * 24.0 + _hour_of_day(rng, n, p_business=0.72)

    fmt_idx = rng.choice(len(PAYMENT_FORMATS), size=n, p=_LAUNDER_FORMAT_P)
    cur_idx = rng.choice(len(CURRENCIES), size=n, p=CURRENCY_P)
    cur, recv_cur, recv_amt = fx_legs(rng, amt, cur_idx)

    return pd.DataFrame(
        {
            "timestamp": start + pd.to_timedelta(base_hour + offsets, unit="h"),
            "from_bank": banks[src],
            "from_account": accounts[src],
            "to_bank": banks[dst],
            "to_account": accounts[dst],
            "amount_paid": amt,
            "payment_currency": cur,
            "amount_received": recv_amt,
            "receiving_currency": recv_cur,
            "payment_format": np.asarray(PAYMENT_FORMATS)[fmt_idx],
            "is_laundering": np.ones(n, dtype=np.int8),
            "typology": pd.array([typology] * n, dtype="string"),
            "pattern_id": pd.array([pattern_id] * n, dtype="Int64"),
            "source": np.full(n, "background", dtype=object),
        }
    )


def _pattern_hops(
    rng: np.random.Generator, typology: str, n_acc: int, total: float
) -> list[tuple[int, int, float]]:
    """Account-index hops realising one instance of ``typology``.

    Each shape matches its name: fan-out sprays from one source, fan-in
    consolidates into one collector, cycle returns to the origin, gather-scatter
    collects then redistributes through a hub, and stack layers sequentially
    through a chain of mules.
    """
    if typology == "fan-out":
        k = int(rng.integers(5, 14))
        src = int(rng.integers(0, n_acc))
        dsts = rng.choice(n_acc, size=k, replace=False)
        tranches = structured_split(rng, total, k)
        return [(src, int(d), float(a)) for d, a in zip(dsts, tranches) if int(d) != src]

    if typology == "fan-in":
        k = int(rng.integers(5, 14))
        dst = int(rng.integers(0, n_acc))
        srcs = rng.choice(n_acc, size=k, replace=False)
        tranches = structured_split(rng, total, k)
        return [(int(s), dst, float(a)) for s, a in zip(srcs, tranches) if int(s) != dst]

    if typology == "cycle":
        k = int(rng.integers(3, 6))
        ring = rng.choice(n_acc, size=k, replace=False).astype(int)
        hops = []
        amt = total
        for i in range(k):
            hops.append((int(ring[i]), int(ring[(i + 1) % k]), float(amt)))
            amt *= rng.uniform(0.97, 0.995)  # skim at each hop
        return hops

    if typology == "gather-scatter":
        k_in = int(rng.integers(4, 9))
        k_out = int(rng.integers(4, 9))
        hub = int(rng.integers(0, n_acc))
        srcs = rng.choice(n_acc, size=k_in, replace=False)
        dsts = rng.choice(n_acc, size=k_out, replace=False)
        in_tranches = structured_split(rng, total, k_in)
        out_tranches = structured_split(rng, total * 0.98, k_out)
        hops = [(int(s), hub, float(a)) for s, a in zip(srcs, in_tranches) if int(s) != hub]
        hops += [(hub, int(d), float(a)) for d, a in zip(dsts, out_tranches) if int(d) != hub]
        return hops

    # stack -- sequential layering through a chain of mules
    k = int(rng.integers(3, 7))
    chain = rng.choice(n_acc, size=k + 1, replace=False).astype(int)
    splits = int(rng.integers(2, 5))
    hops = []
    for i in range(k):
        tranches = structured_split(rng, total * (0.99**i), splits)
        hops += [(int(chain[i]), int(chain[i + 1]), float(a)) for a in tranches]
    return hops


def _inject_patterns(
    rng: np.random.Generator,
    accounts: np.ndarray,
    banks: np.ndarray,
    n_rows_target: int,
    days: int,
    start: pd.Timestamp,
) -> pd.DataFrame:
    """Build labelled laundering instances across the five typologies.

    Cycles through typologies so all five are represented even at a very low
    laundering rate -- per-typology recall is meaningless if a typology has two
    instances in the whole ledger.
    """
    n_acc = len(accounts)
    frames: list[pd.DataFrame] = []
    rows_made = 0
    pattern_id = 0

    while rows_made < n_rows_target:
        typology = _TYPOLOGY_CYCLE[pattern_id % len(_TYPOLOGY_CYCLE)]
        # Snapped to a midnight boundary. _launder_rows adds an hour-of-day on
        # top of this, so an arbitrary fractional base would rotate every
        # pattern's clock by a random amount and wipe out the diurnal structure
        # the hour-of-day draw is there to create -- leaving laundering
        # uniformly spread across the day, which is the easy separator that
        # _hour_of_day exists to avoid.
        base_hour = 24.0 * float(rng.integers(0, max(1, days - 6)))
        # Kept modest so that total/k lands naturally below the reporting
        # threshold for most patterns. Larger totals forced the sub-threshold
        # clip in structured_split to fire on nearly every tranche, which both
        # destroyed the pattern's stated amount and produced an unnaturally
        # tight amount band.
        total = float(rng.uniform(15_000, 150_000))

        hops = _pattern_hops(rng, typology, n_acc, total)
        if not hops:
            pattern_id += 1
            continue

        frames.append(
            _launder_rows(rng, pattern_id, typology, hops, accounts, banks, base_hour, start)
        )
        rows_made += len(hops)
        pattern_id += 1

    return pd.concat(frames, ignore_index=True)


def generate(
    accounts: int = 20_000,
    rows: int = 500_000,
    laundering_rate: float = 0.001,
    start: str = "2026-01-01",
    days: int = 180,
    seed: int = 20260917,
    n_banks: int = 40,
) -> pd.DataFrame:
    """Generate a synthetic ledger in the canonical schema.

    ``laundering_rate`` is the target fraction of rows that are laundering.
    The 0.001 default mirrors the real IBM AML dataset's brutal ~0.1%
    prevalence; developing against a friendlier ratio would hide exactly the
    problems that matter most.
    """
    rng = np.random.default_rng(seed)
    start_ts = pd.Timestamp(start)

    acct_ids = account_ids(accounts)
    acct_banks = rng.integers(0, n_banks, size=accounts).astype(np.int32)

    n_launder = max(1, int(round(rows * laundering_rate)))
    n_background = max(1, rows - n_launder)

    background = _make_background(rng, acct_ids, acct_banks, n_background, days, start_ts)
    laundering = _inject_patterns(rng, acct_ids, acct_banks, n_launder, days, start_ts)

    ledger = pd.concat([background, laundering], ignore_index=True)
    ledger = ledger.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return schema.validate(schema.coerce(ledger))
