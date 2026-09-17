"""Canonical ledger schema.

Every data source -- the synthetic generator, the real IBM AML CSV, PaySim -- is
normalised into this one schema before anything downstream touches it. The
feature pipeline, detector and adversary all speak only this schema.

Two columns extend the raw IBM AML format and matter a great deal:

``typology``
    Which laundering pattern a transaction belongs to (fan-out, fan-in, cycle,
    gather-scatter, stack) or null for benign traffic. This is what makes
    per-typology recall measurable, which objective 1 asks for explicitly.

``pattern_id``
    Groups the transactions that make up a single laundering instance. A
    typology is a property of a *set* of transactions, not of one transaction,
    so without this grouping the labels are much less useful.
"""

from __future__ import annotations

import pandas as pd

#: Laundering typologies this project recognises.
TYPOLOGIES = ("fan-out", "fan-in", "cycle", "gather-scatter", "stack")

#: Payment formats, mirroring the IBM AML dataset's vocabulary. Ordered roughly
#: from lower to higher scrutiny -- the adversary's ``format_mix`` gene biases
#: along this axis.
PAYMENT_FORMATS = (
    "ACH",
    "Cheque",
    "Credit Card",
    "Reinvestment",
    "Wire",
    "Cash",
    "Bitcoin",
)

#: Currencies. Kept small deliberately; cross-currency is a feature signal, and
#: a long tail of currencies would add sparsity without adding insight.
CURRENCIES = ("US Dollar", "Euro", "Rupee", "Yuan", "Bitcoin")

#: Where a row came from. Adversary rows are injected by the GA; keeping them
#: distinguishable is what lets us hold out a provably clean evaluation set.
SOURCES = ("background", "adversary")

#: The structuring threshold the detector watches and the adversary fights.
#: US CTR reporting threshold; the single most important number in the project.
REPORTING_THRESHOLD = 10_000.0

#: column -> dtype. This dict *is* the schema.
DTYPES: dict[str, str] = {
    "timestamp": "datetime64[ns]",
    "from_bank": "int32",
    "from_account": "string",
    "to_bank": "int32",
    "to_account": "string",
    "amount_paid": "float64",
    "payment_currency": "category",
    "amount_received": "float64",
    "receiving_currency": "category",
    "payment_format": "category",
    "is_laundering": "int8",
    "typology": "string",
    "pattern_id": "Int64",
    "source": "category",
}

COLUMNS = tuple(DTYPES)


class SchemaError(ValueError):
    """Raised when a frame does not conform to the canonical ledger schema."""


def empty_ledger() -> pd.DataFrame:
    """An empty frame with correct columns and dtypes.

    Useful as a concatenation base and as a template in tests.
    """
    return pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in DTYPES.items()})


def coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder to canonical column order and cast to canonical dtypes.

    Sources build frames however is convenient, then call this. Raises
    :class:`SchemaError` if required columns are missing, so a source can never
    silently produce a half-formed ledger.
    """
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"missing required column(s): {missing}")

    out = df.loc[:, list(COLUMNS)].copy()
    for col, dtype in DTYPES.items():
        if dtype == "category":
            out[col] = out[col].astype("string").astype("category")
        else:
            out[col] = out[col].astype(dtype)
    return out


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Assert the invariants a ledger must satisfy. Returns ``df`` unchanged.

    Checked here rather than trusted, because every one of these has a plausible
    failure mode: a generator bug, a malformed CSV, or -- most importantly -- an
    adversary decoder that produces impossible transactions. Catching those at
    the schema boundary keeps the rest of the system able to assume sanity.
    """
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"missing required column(s): {missing}")

    if df.empty:
        return df

    if df["amount_paid"].isna().any() or df["amount_received"].isna().any():
        raise SchemaError("amounts must not be null")
    if (df["amount_paid"] <= 0).any() or (df["amount_received"] <= 0).any():
        raise SchemaError("amounts must be strictly positive")
    if df["timestamp"].isna().any():
        raise SchemaError("timestamps must not be null")
    if df["from_account"].isna().any() or df["to_account"].isna().any():
        raise SchemaError("account identifiers must not be null")
    if (df["from_account"] == df["to_account"]).any():
        raise SchemaError("self-transfers are not plausible transactions")

    bad_labels = set(df["is_laundering"].unique()) - {0, 1}
    if bad_labels:
        raise SchemaError(f"is_laundering must be 0/1, saw {bad_labels}")

    seen_typologies = set(df["typology"].dropna().unique())
    unknown = seen_typologies - set(TYPOLOGIES)
    if unknown:
        raise SchemaError(f"unknown typology label(s): {sorted(unknown)}")

    # A typology label without a positive class label (or vice versa) means the
    # ground truth is internally inconsistent, which would quietly corrupt
    # per-typology recall.
    labelled = df["typology"].notna()
    if (df.loc[labelled, "is_laundering"] != 1).any():
        raise SchemaError("rows with a typology must have is_laundering == 1")
    if df.loc[df["is_laundering"] == 1, "typology"].isna().any():
        raise SchemaError("laundering rows must carry a typology label")

    unknown_src = set(df["source"].dropna().unique().tolist()) - set(SOURCES)
    if unknown_src:
        raise SchemaError(f"unknown source value(s): {sorted(unknown_src)}")

    return df


def positive_rate(df: pd.DataFrame) -> float:
    """Fraction of rows labelled laundering. Reported constantly, so given a name."""
    return 0.0 if df.empty else float(df["is_laundering"].mean())
