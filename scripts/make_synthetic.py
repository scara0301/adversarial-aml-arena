"""Generate a synthetic ledger and write it to parquet.

Phase 1, step 1. Lets everything downstream run without the ~500MB Kaggle
download, behind the same loader interface the real data will use.

    uv run python scripts/make_synthetic.py --rows 500000
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from arena.config import Config
from arena.data.loader import save_ledger
from arena.data.synthetic import generate


def main(argv: list[str] | None = None) -> int:
    cfg = Config.load()
    syn = cfg.section("synthetic")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--accounts", type=int, default=syn.get("accounts", 20_000))
    ap.add_argument("--rows", type=int, default=syn.get("rows", 500_000))
    ap.add_argument("--laundering-rate", type=float, default=syn.get("laundering_rate", 0.001))
    ap.add_argument("--days", type=int, default=syn.get("days", 180))
    ap.add_argument("--start", default=syn.get("start", "2026-01-01"))
    ap.add_argument("--seed", type=int, default=syn.get("seed", 20260917))
    ap.add_argument("--out", default=str(cfg.path("paths.synthetic_ledger")))
    args = ap.parse_args(argv)

    print(
        f"generating {args.rows:,} transactions across {args.accounts:,} accounts "
        f"over {args.days} days at a {args.laundering_rate:.3%} laundering rate"
    )

    ledger = generate(
        accounts=args.accounts,
        rows=args.rows,
        laundering_rate=args.laundering_rate,
        start=args.start,
        days=args.days,
        seed=args.seed,
    )

    out = save_ledger(ledger, args.out)

    pos = int(ledger["is_laundering"].sum())
    print(f"\nwrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"  rows            {len(ledger):,}")
    print(f"  laundering      {pos:,} ({pos / len(ledger):.4%})")
    print(f"  patterns        {int(ledger['pattern_id'].nunique()):,}")
    print(f"  accounts        {pd.concat([ledger['from_account'], ledger['to_account']]).nunique():,}")
    print(f"  span            {ledger['timestamp'].min():%Y-%m-%d} .. {ledger['timestamp'].max():%Y-%m-%d}")
    print("\n  transactions by typology")
    counts = ledger["typology"].value_counts(dropna=False)
    for name, count in counts.items():
        label = "benign" if pd.isna(name) else str(name)
        print(f"    {label:<16}{count:>9,}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
