"""Phase 5, step 1: load the real IBM AML data into the canonical schema.

Reads ``data/raw/HI-Small_Trans.csv`` and its ``HI-Small_Patterns.txt``, applies
the three cleaning steps documented in ``arena.data.ibm_aml`` (self-transfers,
the dead tail, unattributed positives), and writes the canonical ledger plus a
data profile. Run once; every later step reads the parquet.

    ARENA_CONFIG=config/ibm.yaml uv run python scripts/prepare_ibm.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from arena.config import Config
from arena.data.loader import load_ledger, save_ledger


def profile(df: pd.DataFrame) -> dict:
    """The facts about the data that the writeup needs, measured not assumed."""
    y = df["is_laundering"] == 1
    amt = df["amount_paid"]
    is_round = (amt % 100.0) < 1e-6
    day = df["timestamp"].dt.floor("D")
    return {
        "rows": int(len(df)),
        "positives": int(y.sum()),
        "positive_rate": float(y.mean()),
        "accounts": int(pd.concat([df["from_account"], df["to_account"]]).nunique()),
        "banks": int(pd.concat([df["from_bank"], df["to_bank"]]).nunique()),
        "start": str(df["timestamp"].min()),
        "end": str(df["timestamp"].max()),
        "days": int(day.nunique()),
        "currencies": int(df["payment_currency"].nunique()),
        "cross_currency_share": float((df["payment_currency"].astype(str) != df["receiving_currency"].astype(str)).mean()),
        "round_100_share": {"benign": float(is_round[~y].mean()), "laundering": float(is_round[y].mean())},
        "median_amount": {"benign": float(amt[~y].median()), "laundering": float(amt[y].median())},
        "typology_rows": {k: int(v) for k, v in df.loc[y, "typology"].value_counts().items()},
        "typology_patterns": {k: int(v) for k, v in df.loc[y].groupby("typology")["pattern_id"].nunique().items()},
        "payment_format_share": {
            "benign": {k: float(v) for k, v in df.loc[~y, "payment_format"].value_counts(normalize=True).items()},
            "laundering": {k: float(v) for k, v in df.loc[y, "payment_format"].value_counts(normalize=True).items()},
        },
        "rows_per_day": {str(k.date()): int(v) for k, v in day.value_counts().sort_index().items()},
        "positives_per_day": {str(k.date()): int(v) for k, v in day[y].value_counts().sort_index().items()},
    }


def main(argv: list[str] | None = None) -> int:
    cfg = Config.load()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trans", default=str(cfg.path("ibm.trans")) if cfg.get("ibm.trans") else "data/raw/HI-Small_Trans.csv")
    ap.add_argument("--out", default=str(cfg.path("paths.synthetic_ledger")))
    ap.add_argument("--reports", default=str(cfg.path("paths.reports")))
    args = ap.parse_args(argv)
    if "ibm" not in str(args.out):
        print("refusing to overwrite the synthetic ledger: run with ARENA_CONFIG=config/ibm.yaml")
        return 2

    t0 = time.time()
    df = load_ledger("ibm", path=args.trans)
    cleaning = df.attrs.get("cleaning", {})
    print(f"loaded {len(df):,} rows in {time.time() - t0:.0f}s")
    for k, v in cleaning.items():
        print(f"   {k:<36} {v:,}" if isinstance(v, int) else f"   {k:<36} {v}")

    out = save_ledger(df, args.out)
    prof = profile(df)
    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / "phase5_data.json"
    path.write_text(json.dumps({"source": Path(args.trans).name, "cleaning": cleaning, "profile": prof},
                               indent=2), encoding="utf-8")

    print(f"\n{prof['rows']:,} rows, {prof['positives']:,} positives ({prof['positive_rate']:.3%}), "
          f"{prof['accounts']:,} accounts, {prof['days']} days, {prof['currencies']} currencies")
    print("typology rows:", prof["typology_rows"])
    print(f"round-$100 share: benign {prof['round_100_share']['benign']:.4%}, "
          f"laundering {prof['round_100_share']['laundering']:.4%}")
    print(f"\nwrote {out} and {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
