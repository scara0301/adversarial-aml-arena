"""Reader for the real IBM AML dataset (Kaggle: ealtman2019).

Two files per variant, and both are needed:

``HI-Small_Trans.csv``
    The transactions. Columns: ``Timestamp``, ``From Bank``, ``Account``,
    ``To Bank``, ``Account.1``, ``Amount Received``, ``Receiving Currency``,
    ``Amount Paid``, ``Payment Currency``, ``Payment Format``, ``Is Laundering``.
    Note the duplicated ``Account`` header, which pandas disambiguates to
    ``Account`` / ``Account.1``.

``HI-Small_Patterns.txt``
    Blocks of the form ``BEGIN LAUNDERING ATTEMPT - <TYPE>`` / ``END ...``,
    each listing the member transactions of one laundering instance. The
    transactions CSV alone gives a binary label; only this file gives the
    *typology*, which objective 1 requires. Parsing it is therefore not
    optional.

This module is written now, ahead of Phase 5, so that the loader contract is
real rather than aspirational -- but it is unexercised until the download
happens.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from arena import schema

_COLUMN_MAP = {
    "Timestamp": "timestamp",
    "From Bank": "from_bank",
    "Account": "from_account",
    "To Bank": "to_bank",
    "Account.1": "to_account",
    "Amount Paid": "amount_paid",
    "Payment Currency": "payment_currency",
    "Amount Received": "amount_received",
    "Receiving Currency": "receiving_currency",
    "Payment Format": "payment_format",
    "Is Laundering": "is_laundering",
}

# The dataset's pattern names, mapped onto our typology vocabulary. SCATTER-
# GATHER and BIPARTITE are folded into their nearest equivalents; RANDOM has no
# recognisable shape, so it is labelled "stack" (generic layering) rather than
# being silently dropped.
_TYPOLOGY_MAP = {
    "FAN-OUT": "fan-out",
    "FAN-IN": "fan-in",
    "CYCLE": "cycle",
    "GATHER-SCATTER": "gather-scatter",
    "SCATTER-GATHER": "gather-scatter",
    "BIPARTITE": "fan-out",
    "STACK": "stack",
    "RANDOM": "stack",
}

_BEGIN = re.compile(r"BEGIN LAUNDERING ATTEMPT\s*-\s*(?P<kind>[A-Z\-]+)")
_END = re.compile(r"END LAUNDERING ATTEMPT")


def parse_patterns(path: str | Path) -> pd.DataFrame:
    """Parse ``*_Patterns.txt`` into a join key table.

    Returns a frame of ``[timestamp, from_account, to_account, amount_paid,
    typology, pattern_id]``. The first four columns form the join key back onto
    the transactions CSV -- the patterns file has no transaction id, so the
    transaction's own fields are the only way to match.
    """
    rows: list[dict] = []
    pattern_id = -1
    kind: str | None = None

    with Path(path).open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue

            begin = _BEGIN.search(line)
            if begin:
                pattern_id += 1
                kind = _TYPOLOGY_MAP.get(begin.group("kind").upper(), "stack")
                continue
            if _END.search(line):
                kind = None
                continue
            if kind is None:
                continue

            # Member transaction lines share the CSV's comma layout.
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 10:
                continue
            try:
                rows.append(
                    {
                        "timestamp": pd.to_datetime(parts[0], errors="raise"),
                        "from_account": parts[2],
                        "to_account": parts[4],
                        "amount_paid": float(parts[7]),
                        "typology": kind,
                        "pattern_id": pattern_id,
                    }
                )
            except (ValueError, TypeError):
                # A header or comment line inside a block; skip rather than fail
                # the whole parse.
                continue

    return pd.DataFrame(rows)


def load_ibm_aml(
    trans_path: str | Path,
    patterns_path: str | Path | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Load the IBM AML transactions CSV into the canonical schema.

    Args:
        trans_path: Path to ``*_Trans.csv``.
        patterns_path: Path to the matching ``*_Patterns.txt``. Defaults to the
            sibling file with ``_Trans.csv`` replaced by ``_Patterns.txt``.
        nrows: Read only the first N rows. HI-Small is ~5M rows / ~500MB, so
            this is the usual way to work interactively.
    """
    trans_path = Path(trans_path)
    df = pd.read_csv(trans_path, nrows=nrows)

    missing = [c for c in _COLUMN_MAP if c not in df.columns]
    if missing:
        raise ValueError(f"{trans_path.name} is missing expected column(s): {missing}")

    df = df.rename(columns=_COLUMN_MAP)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["from_account"] = df["from_account"].astype("string")
    df["to_account"] = df["to_account"].astype("string")
    df["typology"] = pd.array([None] * len(df), dtype="string")
    df["pattern_id"] = pd.array([None] * len(df), dtype="Int64")
    df["source"] = "background"

    if patterns_path is None:
        candidate = trans_path.with_name(trans_path.name.replace("_Trans.csv", "_Patterns.txt"))
        patterns_path = candidate if candidate.exists() else None

    if patterns_path is not None:
        pats = parse_patterns(patterns_path)
        if not pats.empty:
            key = ["timestamp", "from_account", "to_account", "amount_paid"]
            pats["from_account"] = pats["from_account"].astype("string")
            pats["to_account"] = pats["to_account"].astype("string")
            df = df.drop(columns=["typology", "pattern_id"]).merge(
                pats.drop_duplicates(subset=key), on=key, how="left"
            )
            df["typology"] = df["typology"].astype("string")
            df["pattern_id"] = df["pattern_id"].astype("Int64")

    # Reconcile the two label sources. The binary flag and the patterns file can
    # disagree at the margins; schema.validate() requires them to be consistent,
    # so resolve here rather than letting validation fail on real data.
    #
    # A positive flag with no matched pattern gets the generic "stack" label; a
    # matched pattern with a zero flag is promoted to positive.
    flagged = df["is_laundering"].astype("int8") == 1
    df.loc[flagged & df["typology"].isna(), "typology"] = "stack"
    df.loc[df["typology"].notna(), "is_laundering"] = 1

    return schema.validate(schema.coerce(df))
