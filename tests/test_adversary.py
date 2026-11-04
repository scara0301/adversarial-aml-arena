"""Adversary tests, including the invariant-2 conservation check.

``test_money_is_conserved_under_crossover`` is the counterpart to the feature
equivalence test. It asserts the property over a large sample of *arbitrary*
genomes and crossover offspring rather than over genomes the GA happens to
produce, because the claim in CLAUDE.md is that conservation holds by
construction -- and a property that holds by construction should survive inputs
no sane search would generate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from arena.adversary.constraints import (
    ImplausibleStrategy,
    conservation_error,
    repair_genome,
    validate_sub_ledger,
)
from arena.adversary.decoder import (
    MAX_TRANSACTIONS,
    MIN_GAP_HOURS,
    MIN_TRANCHE,
    decode,
    delivered_amount,
    elapsed_hours,
)
from arena.adversary.fitness import FitnessConfig, Outcome, evaluate_episode
from arena.adversary.ga import naive_genome
from arena.adversary.genome import (
    GENE_INDEX,
    HIGH,
    LOW,
    N_GENES,
    decode_genes,
    distance,
    random_genome,
)
from arena.schema import REPORTING_THRESHOLD


# --------------------------------------------------------------------- genome


def test_bounds_are_respected_after_repair():
    wild = np.array([-500.0, 99.0, -3.0, 5.0, -1.0, 1e6, 7.0, -2.0, 50.0, 3.0, -9.0, 1e4])
    fixed = repair_genome(wild)
    assert (fixed >= LOW).all()
    assert (fixed <= HIGH).all()


def test_integer_genes_decode_to_ints():
    p = decode_genes(random_genome(np.random.default_rng(3)))
    for name in ("typology_template", "n_hops", "n_splits", "mule_pool_size"):
        assert isinstance(p[name], int)


def test_distance_is_a_metric_on_the_unit_scale():
    rng = np.random.default_rng(0)
    a, b = random_genome(rng), random_genome(rng)
    assert distance(a, a) == pytest.approx(0.0)
    assert distance(a, b) == pytest.approx(distance(b, a))
    assert 0.0 <= distance(a, b) <= 1.5


# ------------------------------------------------------------------- decoder


def test_money_is_conserved_under_crossover(episode):
    """Invariant 2, over 600 arbitrary crossover offspring.

    Crossover is the operator that breaks naive designs: splicing two parents at
    a random point produces gene combinations neither parent had. A design that
    stored tranche *amounts* in the genome would leak or invent money here. This
    one cannot, because no gene holds an absolute amount -- the decoder always
    allocates exactly the balance it was handed.
    """
    worst = 0.0
    checked = 0
    for i in range(600):
        r1 = np.random.default_rng(i)
        r2 = np.random.default_rng(i + 500_000)
        a, b = random_genome(r1), random_genome(r2)
        cut = int(r1.integers(1, N_GENES))
        child = np.concatenate([a[:cut], b[cut:]])

        sub = decode(child, episode, np.random.default_rng(i))
        if sub.empty:
            continue
        checked += 1
        worst = max(worst, conservation_error(sub, episode))

    assert checked > 500, "most random genomes should decode to a real strategy"
    # Absolute error in dollars on a $250,000 goal: 1e-6 is ~4e-12 relative.
    assert worst < 1e-6, f"INVARIANT 2 VIOLATED: conservation error ${worst}"


def test_all_random_strategies_are_plausible(episode):
    """The population is always inside the plausibility envelope (hard repair).

    Catches, in particular, an account spending money before it received it --
    which would be the adversary exploiting a simulation bug rather than
    discovering an evasion.
    """
    failures = []
    for i in range(600):
        genome = random_genome(np.random.default_rng(i))
        sub = decode(genome, episode, np.random.default_rng(i))
        if sub.empty:
            continue
        try:
            validate_sub_ledger(sub, episode)
        except ImplausibleStrategy as exc:
            failures.append(f"genome {i}: {exc}")
    assert not failures, "implausible strategies:\n" + "\n".join(failures[:10])


def test_tranches_respect_the_plausibility_floor(episode):
    for i in range(200):
        sub = decode(random_genome(np.random.default_rng(i)), episode, np.random.default_rng(i))
        if sub.empty:
            continue
        assert sub["amount_paid"].min() >= MIN_TRANCHE - 1e-6


def test_transaction_count_is_capped(episode):
    for i in range(200):
        sub = decode(random_genome(np.random.default_rng(i)), episode, np.random.default_rng(i))
        assert len(sub) <= MAX_TRANSACTIONS


def test_threshold_hug_controls_tranche_size(episode):
    """The gene must span "one big transfer" to "sit under the threshold".

    A regression guard. An earlier linear interpolation bottomed out at a $5,000
    target, so every genome structured automatically -- the gene had no range and
    the naive baseline was already evading the threshold feature.
    """
    rng = np.random.default_rng(1)

    low = random_genome(rng).copy()
    low[GENE_INDEX["threshold_hug"]] = 0.0
    low[GENE_INDEX["n_splits"]] = 1
    low[GENE_INDEX["n_hops"]] = 1
    sub_low = decode(low, episode, np.random.default_rng(1))

    high = low.copy()
    high[GENE_INDEX["threshold_hug"]] = 1.0
    sub_high = decode(high, episode, np.random.default_rng(1))

    assert sub_low["amount_paid"].max() > REPORTING_THRESHOLD, "hug=0 must not structure"
    assert sub_high["amount_paid"].max() < REPORTING_THRESHOLD, "hug=1 must stay sub-threshold"
    assert len(sub_high) > len(sub_low)


def test_no_account_sends_twice_at_the_same_instant(episode):
    """Regression guard for the simulation exploit the GA found.

    With mean_delay_h at 0 the decoder used to emit every tranche at one
    identical timestamp. Because the as-of feature queries exclude ties, none of
    those transactions could see the others, every burst feature read zero, and
    the detector's recall on the strategy was exactly 0.000 -- a perfect score
    obtained by exploiting the simulation rather than by evading anything.

    Tested at the extreme (all timing genes zeroed) and across random genomes.
    """
    from arena.features.account_graph import to_hours

    zero_delay = random_genome(np.random.default_rng(0)).copy()
    zero_delay[GENE_INDEX["mean_delay_h"]] = 0.0
    zero_delay[GENE_INDEX["delay_jitter"]] = 0.0
    zero_delay[GENE_INDEX["dormancy_h"]] = 0.0
    zero_delay[GENE_INDEX["n_splits"]] = 12
    zero_delay[GENE_INDEX["business_hours_bias"]] = 1.0

    candidates = [zero_delay] + [random_genome(np.random.default_rng(i)) for i in range(200)]
    for k, genome in enumerate(candidates):
        sub = decode(genome, episode, np.random.default_rng(k))
        if sub.empty:
            continue
        hours = to_hours(sub["timestamp"], episode.epoch)
        frame = pd.DataFrame({"sender": sub["from_account"].to_numpy(), "h": hours})
        gaps = frame.sort_values("h").groupby("sender", observed=True)["h"].diff().dropna()
        if len(gaps):
            assert float(gaps.min()) >= MIN_GAP_HOURS - 1e-9, (
                f"genome {k}: two transfers from one account "
                f"{float(gaps.min()) * 3600:.4f}s apart"
            )


def test_adversary_rows_carry_no_currency_or_fx_fingerprint(episode):
    """Regression guard for a fingerprint the first closed-loop run exposed.

    The decoder used to charge a fixed 0.2% fee and always pay in USD, so every
    adversary row had ``fx_loss`` of exactly 0.002 while benign traffic has 0 (93%
    of rows) or a 0.5-3% spread. A detector retrained on decoded rows learned that
    constant instead of the strategy -- about a quarter of its apparent catch rate
    on the round-1 adversary, and it leaned on ``fx_loss`` ten times harder than the
    round-0 detector. Currency and FX now come from the same shared process every
    other row uses, so these distributions must match the background's.
    """
    rows = []
    for i in range(300):
        sub = decode(random_genome(np.random.default_rng(i)), episode, np.random.default_rng(i))
        if not sub.empty:
            rows.append(sub)
    sub = pd.concat(rows, ignore_index=True)
    assert len(sub) > 2000

    fx = 1.0 - sub["amount_received"] / sub["amount_paid"]
    zero = float((fx.abs() < 1e-9).mean())
    assert 0.88 < zero < 0.97, f"{zero:.1%} of adversary rows have zero FX loss; background has ~93%"

    nonzero = fx[fx.abs() > 1e-9].round(6)
    assert nonzero.value_counts(normalize=True).iloc[0] < 0.05, "a single FX-loss value dominates"
    assert nonzero.min() > 0.0049 and nonzero.max() < 0.0301, "spread outside the shared process's range"

    cross = float((sub["payment_currency"].astype(str) != sub["receiving_currency"].astype(str)).mean())
    assert 0.04 < cross < 0.10, f"cross-currency rate {cross:.1%}; background is ~7%"
    assert sub["payment_currency"].astype(str).nunique() >= 4, "adversary rows must not all share one currency"


def test_decoder_is_deterministic_given_a_seed(episode):
    genome = random_genome(np.random.default_rng(11))
    a = decode(genome, episode, np.random.default_rng(4))
    b = decode(genome, episode, np.random.default_rng(4))
    assert a.equals(b)


def test_timestamps_never_precede_the_episode_start(episode):
    from arena.features.account_graph import to_hours

    for i in range(100):
        sub = decode(random_genome(np.random.default_rng(i)), episode, np.random.default_rng(i))
        if sub.empty:
            continue
        hours = to_hours(sub["timestamp"], episode.epoch)
        assert hours.min() >= episode.start_hour - 1e-6


def test_delivered_amount_excludes_post_deadline_transfers(episode):
    """The deadline must actually bite, or stalling becomes a free evasion."""
    from dataclasses import replace

    genome = naive_genome()
    sub = decode(genome, episode, np.random.default_rng(2))
    assert delivered_amount(sub, episode) > 0

    impossible = replace(episode, deadline_hours=0.0)
    assert delivered_amount(sub, impossible) == 0.0


def test_naive_strategy_does_not_structure():
    """The baseline must be genuinely unsophisticated, or the Phase 2 headline
    comparison is measuring nothing."""
    p = decode_genes(naive_genome())
    assert p["threshold_hug"] == 0.0
    assert p["business_hours_bias"] == 0.0
    assert p["n_hops"] == 1


# ------------------------------------------------------------------- fitness


class _StubDetector:
    """Detector stand-in with a fixed score, so fitness can be tested alone."""

    def __init__(self, score: float) -> None:
        self.score = score
        self.window_hours = (1.0, 24.0, 168.0)
        self.features: list[str] = []

    def score_ledger(self, ledger, graph):  # noqa: ARG002
        return np.full(len(ledger), self.score, dtype=float)


def test_zero_completion_gives_zero_fitness(episode, graph, rng):
    """The "move nothing" degenerate optimum must be worthless.

    With an additive fitness this strategy is optimal: no transactions means
    nothing is flagged, so evasion is a perfect 1.0.
    """
    from dataclasses import replace

    undetected = _StubDetector(0.0)  # never flagged -> evasion 1.0
    impossible = replace(episode, deadline_hours=0.0)  # nothing can be delivered

    out = evaluate_episode(naive_genome(), impossible, undetected, graph, 0.5, rng)
    assert out.evasion == pytest.approx(1.0), "nothing was flagged"
    assert out.completion == pytest.approx(0.0)
    assert out.fitness == 0.0, "perfect evasion with zero completion must score 0"


def test_completion_floor_is_a_hard_cliff(episode, graph, rng):
    cfg = FitnessConfig(completion_floor=0.99)
    undetected = _StubDetector(0.0)
    out = evaluate_episode(naive_genome(), episode, undetected, graph, 0.5, rng, cfg)
    # The naive strategy loses a little to fees, landing just under a 0.99 floor.
    if out.completion < 0.99:
        assert out.fitness == 0.0


def test_evasion_is_amount_weighted(episode, graph, rng):
    """Flagging must cost in proportion to the money flagged.

    Under an unweighted flag rate, one enormous transfer is the best strategy
    because it minimises the number of flag opportunities.
    """
    flagged = _StubDetector(1.0)
    out = evaluate_episode(naive_genome(), episode, flagged, graph, 0.5, rng)
    assert out.evasion == pytest.approx(0.0), "everything flagged -> no evasion"
    assert out.fitness == 0.0

    clean = _StubDetector(0.0)
    out = evaluate_episode(naive_genome(), episode, clean, graph, 0.5, rng)
    assert out.evasion == pytest.approx(1.0)
    assert out.fitness > 0.0


def test_fitness_is_never_negative(episode, graph, rng):
    """A negative fitness would break tournament selection's assumptions."""
    expensive = FitnessConfig(gamma=100.0)
    for i in range(50):
        out = evaluate_episode(
            random_genome(np.random.default_rng(i)), episode, _StubDetector(0.0),
            graph, 0.5, rng, expensive,
        )
        assert out.fitness >= 0.0


def test_outcome_reports_consistent_counts(episode, graph, rng):
    out = evaluate_episode(naive_genome(), episode, _StubDetector(1.0), graph, 0.5, rng)
    assert isinstance(out, Outcome)
    assert out.n_flagged <= out.n_transactions
    assert out.n_transactions == len(out.sub_ledger)


def test_elapsed_hours_is_non_negative(episode):
    sub = decode(naive_genome(), episode, np.random.default_rng(0))
    assert elapsed_hours(sub, episode) >= 0.0
