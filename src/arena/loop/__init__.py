"""Phase 3: the closed loop -- detector and adversary retrained against each other."""

from arena.loop.evaluation import StrategyScore, score_pool, score_strategy
from arena.loop.orchestrator import ArenaResult, ArenaSettings, fingerprint, run_arena
from arena.loop.pool import Materialized, StrategyPool

__all__ = [
    "ArenaResult",
    "ArenaSettings",
    "Materialized",
    "StrategyPool",
    "StrategyScore",
    "fingerprint",
    "run_arena",
    "score_pool",
    "score_strategy",
]
