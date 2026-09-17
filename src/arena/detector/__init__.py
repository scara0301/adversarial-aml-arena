"""AML detector: model, threshold policy, and imbalance-aware metrics."""

from arena.detector.metrics import evaluate, format_report
from arena.detector.model import Detector, Split, time_split, train
from arena.detector.threshold import threshold_for_budget

__all__ = [
    "Detector",
    "Split",
    "time_split",
    "train",
    "threshold_for_budget",
    "evaluate",
    "format_report",
]
