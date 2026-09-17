"""Single entry point for every data source.

Everything downstream calls :func:`load_ledger` and receives a validated
canonical frame. Swapping synthetic data for the real IBM AML CSV must change
nothing above this line -- that is the whole point of routing through here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from arena import schema


def load_ledger(
    source: str = "synthetic",
    path: str | Path | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Load a ledger in the canonical schema.

    Args:
        source: ``"synthetic"`` generates data in memory; ``"parquet"`` reads a
            previously generated ledger; ``"ibm"`` reads the real Kaggle CSV;
            ``"paysim"`` reads PaySim (Phase 5, partial feature support only).
        path: File path, required for every source except ``"synthetic"``.
        **kwargs: Forwarded to the underlying source.

    Returns:
        A validated canonical ledger frame.
    """
    if source == "synthetic":
        from arena.data.synthetic import generate

        return generate(**kwargs)

    if source == "parquet":
        if path is None:
            raise ValueError("source='parquet' requires a path")
        df = pd.read_parquet(path)
        return schema.validate(schema.coerce(df))

    if source == "ibm":
        from arena.data.ibm_aml import load_ibm_aml

        if path is None:
            raise ValueError("source='ibm' requires a path to *_Trans.csv")
        return load_ibm_aml(path, **kwargs)

    if source == "paysim":
        raise NotImplementedError(
            "PaySim support is Phase 5. Its schema has no comparable account graph or "
            "typology labels, so only the per-transaction feature subset will transfer."
        )

    raise ValueError(f"unknown source {source!r}")


def save_ledger(df: pd.DataFrame, path: str | Path) -> Path:
    """Validate and write a ledger to parquet, creating parent directories."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    schema.validate(df).to_parquet(out, index=False)
    return out
