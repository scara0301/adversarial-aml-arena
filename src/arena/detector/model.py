"""LightGBM AML detector, with the time-based split and frozen holdout.

Two things here are load-bearing beyond the model itself.

**The split is time-based, never random.** A random split puts transactions from
the same laundering pattern -- and the same account histories -- on both sides of
the boundary. Since account features are cumulative over time, a random split
lets the model see an account's later behaviour while scoring its earlier
transactions. Every metric would be inflated, and the inflation would be
invisible. Chronological splitting is the only honest option for a temporal
graph problem.

**The frozen clean holdout is carved out here** (invariant 4), in Phase 1, before
the adversary exists. It is the last chronological slice, written to parquet, and
it is never augmented with adversary data in any later round. All cross-round
metrics are computed on it, which is what makes the arms-race curve trustworthy:
if the evaluation set drifted as the adversary contributed to it, a rising recall
could just mean the adversary's own transactions got easier.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from arena.features import AccountGraph, featurize
from arena.features.pipeline import DEFAULT_WINDOWS

#: Below this many positives in the validation split, early stopping is disabled
#: because the PR-AUC signal is too noisy to stop on. See ``train``.
MIN_VALID_POSITIVES_FOR_EARLY_STOP = 100

DEFAULT_PARAMS = {
    "objective": "binary",
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 40,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "verbose": -1,
}


@dataclass(slots=True)
class Split:
    """Chronological three-way split of a ledger.

    ``holdout`` is the frozen clean evaluation set (invariant 4).
    """

    train: pd.DataFrame
    valid: pd.DataFrame
    holdout: pd.DataFrame

    def summary(self) -> str:
        def one(name: str, df: pd.DataFrame) -> str:
            pos = int(df["is_laundering"].sum())
            span = f"{df['timestamp'].min():%Y-%m-%d} .. {df['timestamp'].max():%Y-%m-%d}"
            return f"  {name:<9}{len(df):>9,} rows  {pos:>6,} pos ({pos/max(len(df),1):.4%})  {span}"

        return "\n".join(
            ["chronological split", one("train", self.train), one("valid", self.valid), one("holdout", self.holdout)]
        )


def downsample_negatives(
    df: pd.DataFrame, ratio: float = 25.0, seed: int = 0
) -> pd.DataFrame:
    """Keep every positive and ``ratio`` negatives per positive.

    Standard practice for extreme imbalance, and it does two things for us.

    It makes training tractable: at a 0.2% positive rate, 2M rows carry ~4,000
    positives, and the other 1.99M negatives contribute almost nothing per row
    while costing everything in memory and time. A 25:1 sample keeps the
    informative rows and drops the redundant ones.

    More subtly, it lets the *graph* stay complete while the *training set*
    shrinks. Account features are computed against the full ledger, so an
    account's degree and volume history are unaffected by which of its
    transactions we happen to train on. Downsampling the rows would corrupt the
    features if the graph were built from the sample -- it is not.

    Calibration shifts, but that is harmless here: the decision threshold is a
    quantile of scores on the holdout (invariant 3), so only the *ranking* of
    scores matters, never their absolute values.

    Applied to train and valid only. The holdout keeps its true prevalence so
    that reported metrics describe the real operating regime.
    """
    pos = df[df["is_laundering"] == 1]
    neg = df[df["is_laundering"] == 0]
    n_keep = min(len(neg), int(len(pos) * ratio))
    if n_keep >= len(neg):
        return df
    kept = neg.sample(n=n_keep, random_state=seed)
    return pd.concat([pos, kept]).sort_values("timestamp", kind="mergesort")


def time_split(
    ledger: pd.DataFrame,
    train_frac: float = 0.60,
    valid_frac: float = 0.15,
) -> Split:
    """Split chronologically into train / valid / frozen holdout.

    The remainder after ``train_frac`` and ``valid_frac`` becomes the holdout.
    """
    if train_frac + valid_frac >= 1.0:
        raise ValueError("train_frac + valid_frac must leave room for a holdout")

    ordered = ledger.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    n = len(ordered)
    i_train = int(n * train_frac)
    i_valid = int(n * (train_frac + valid_frac))

    return Split(
        train=ordered.iloc[:i_train].copy(),
        valid=ordered.iloc[i_train:i_valid].copy(),
        holdout=ordered.iloc[i_valid:].copy(),
    )


class Detector:
    """A trained AML classifier plus everything needed to reproduce its scoring.

    Carries the feature column order, the window sizes and the graph epoch
    alongside the booster. Those are not incidental: the GA scores candidates
    against this object thousands of times, and a silent feature-order mismatch
    would produce plausible-looking nonsense rather than an error.
    """

    __slots__ = ("booster", "features", "window_hours", "epoch", "params")

    def __init__(
        self,
        booster: lgb.Booster,
        features: list[str],
        window_hours: tuple[float, ...],
        epoch: pd.Timestamp,
        params: dict | None = None,
    ) -> None:
        self.booster = booster
        self.features = features
        self.window_hours = window_hours
        self.epoch = epoch
        self.params = params or {}

    # ------------------------------------------------------------------ scoring

    def score_features(self, X: pd.DataFrame) -> np.ndarray:
        """Score an already-built feature frame.

        Reindexes to the training column order and fills anything absent with
        zero, so a caller cannot silently shift the columns.
        """
        aligned = X.reindex(columns=self.features, fill_value=0.0)
        return self.booster.predict(aligned.to_numpy(dtype=np.float64))

    def score_ledger(self, ledger: pd.DataFrame, graph: AccountGraph) -> np.ndarray:
        """Featurize and score transactions.

        This is the method the GA fitness function calls, with ``graph`` being an
        overlay. It goes through the same ``featurize`` as training (invariant 1).
        """
        if ledger.empty:
            return np.zeros(0, dtype=np.float64)
        return self.score_features(featurize(ledger, graph, self.window_hours))

    # ---------------------------------------------------------- persistence

    def save(self, path: str | Path) -> Path:
        """Write the booster natively plus a sidecar of everything else."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(out))
        meta = {
            "features": self.features,
            "window_hours": list(self.window_hours),
            "epoch": str(self.epoch),
            "params": self.params,
        }
        out.with_suffix(out.suffix + ".meta.json").write_text(json.dumps(meta, indent=2))
        return out

    @classmethod
    def load(cls, path: str | Path) -> Detector:
        p = Path(path)
        meta = json.loads(p.with_suffix(p.suffix + ".meta.json").read_text())
        return cls(
            booster=lgb.Booster(model_file=str(p)),
            features=meta["features"],
            window_hours=tuple(meta["window_hours"]),
            epoch=pd.Timestamp(meta["epoch"]),
            params=meta.get("params", {}),
        )

    # -------------------------------------------------------------- inspection

    def importance(self, top: int = 20) -> pd.Series:
        """Gain-based feature importance.

        Used in the phase writeups and, in Phase 3, to show *which* features the
        adversary forces the detector to stop trusting -- the honest substitute
        for the synopsis's "shifting decision boundary".
        """
        gains = self.booster.feature_importance(importance_type="gain")
        return pd.Series(gains, index=self.features).sort_values(ascending=False).head(top)


def train(
    split: Split,
    graph: AccountGraph,
    params: dict | None = None,
    window_hours: tuple[float, ...] = DEFAULT_WINDOWS,
    sample_weight_col: str | None = None,
    negative_ratio: float | None = 25.0,
    seed: int = 0,
) -> tuple[Detector, dict[str, pd.DataFrame]]:
    """Train a detector on a chronological split.

    ``scale_pos_weight`` is set from the observed imbalance rather than tuned.
    At ~1000:1 the gradient signal from the positive class is otherwise
    negligible and the model collapses to predicting the majority.

    Args:
        split: Chronological split; only ``train`` and ``valid`` are used here.
        graph: Account graph covering the whole ledger.
        params: LightGBM overrides.
        window_hours: Rolling window sizes, recorded on the detector.
        sample_weight_col: Optional column in ``split.train`` holding per-row
            weights. Phase 3 uses this to down-weight injected adversary rows.

    Returns:
        The trained detector, and the feature frames keyed by split name (kept
        so callers can evaluate without re-featurizing).
    """
    # Downsample the rows we train on, but featurize them against the full
    # graph the caller passed. The graph is never downsampled -- see
    # downsample_negatives.
    train_rows = split.train
    valid_rows = split.valid
    if negative_ratio is not None:
        train_rows = downsample_negatives(train_rows, negative_ratio, seed)
        valid_rows = downsample_negatives(valid_rows, negative_ratio, seed)

    X_train = featurize(train_rows, graph, window_hours)
    X_valid = featurize(valid_rows, graph, window_hours)
    y_train = train_rows["is_laundering"].to_numpy(dtype=int)
    y_valid = valid_rows["is_laundering"].to_numpy(dtype=int)

    weights = (
        train_rows[sample_weight_col].to_numpy(dtype=float)
        if sample_weight_col and sample_weight_col in train_rows.columns
        else None
    )

    detector = fit_detector(
        X_train, y_train, X_valid, y_valid,
        window_hours=window_hours, epoch=graph.epoch, params=params, weights=weights,
    )
    return detector, {"train": X_train, "valid": X_valid}


def fit_detector(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    *,
    window_hours: tuple[float, ...],
    epoch: pd.Timestamp,
    params: dict | None = None,
    weights: np.ndarray | None = None,
) -> Detector:
    """Fit a detector on pre-built feature matrices.

    Split out of :func:`train` for Phase 3. The closed loop retrains the
    detector every round on the original training rows *plus* adversary rows,
    and the adversary rows are featurized through the graph overlay (the same
    path the GA scores them by) rather than through the batch path ``train``
    uses. Building the matrices is therefore the caller's job; the fitting
    policy -- imbalance weighting, PR-AUC early stopping -- stays in exactly
    one place, so round 0 and round N are trained by identical code.
    """
    cfg = {**DEFAULT_PARAMS, **(params or {})}

    n_pos = max(int((y_train == 1).sum()), 1)
    n_neg = int((y_train == 0).sum())
    cfg["scale_pos_weight"] = n_neg / n_pos

    clf = lgb.LGBMClassifier(**cfg)
    fit_kwargs = {}
    n_valid_pos = int(y_valid.sum()) if len(y_valid) else 0

    # Early stopping needs enough positives for the validation metric to be
    # stable. Below this, PR-AUC swings wildly from one tree to the next and
    # early stopping latches onto a lucky first iteration -- we measured it
    # stopping at a single tree with 24 validation positives. In that regime
    # training the full schedule is strictly better than stopping on noise.
    if n_valid_pos >= MIN_VALID_POSITIVES_FOR_EARLY_STOP:
        # Stop on PR-AUC, not log-loss: at this prevalence log-loss is dominated
        # by the negatives and flattens long before the ranking of the positives
        # stops improving.
        fit_kwargs = {
            "eval_set": [(X_valid, y_valid)],
            "eval_metric": "average_precision",
            "callbacks": [lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        }

    clf.fit(X_train, y_train, sample_weight=weights, **fit_kwargs)

    return Detector(
        booster=clf.booster_,
        features=list(X_train.columns),
        window_hours=window_hours,
        epoch=epoch,
        params=cfg,
    )
