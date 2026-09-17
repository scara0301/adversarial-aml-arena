"""The plausibility envelope, enforced by hard repair.

Objective 2 requires the adversary's actions to be "constrained to plausible
transaction behaviour". There are two ways to do that, and the choice matters.

**Penalty terms** subtract from fitness when a strategy is implausible. The
problem is that fitness then measures a blend of evasion skill and constraint
violation, and the two cannot be separated. A genome scoring 0.4 might be a
brilliant evader that broke one rule, or a mediocre one that broke none. Worse,
the GA spends its search budget learning to be feasible instead of learning to
evade, and reported "evasion" numbers become uninterpretable.

**Hard repair** -- used here -- projects every genome into the feasible region
before it is ever evaluated. The entire population is always plausible, so
fitness measures evasion and nothing else, and every generation of search
pressure goes into the actual objective.

Repair happens at two levels:

* :func:`repair_genome` clamps genes to their declared bounds. Cheap, and it
  runs after every crossover and mutation.
* :func:`validate_sub_ledger` checks the decoded transactions -- the properties
  that cannot be expressed as gene bounds, such as causality and monotonic
  per-account timestamps. This one is an assertion rather than a repair: if the
  decoder produced an impossible sub-ledger, that is a bug to fix, not
  something to paper over at evaluation time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from arena.adversary.decoder import MAX_TRANSACTIONS, MIN_GAP_HOURS, MIN_TRANCHE, Episode
from arena.adversary.genome import HIGH, LOW
from arena.features.account_graph import to_hours


class ImplausibleStrategy(AssertionError):
    """The decoder emitted a sub-ledger that could not occur in reality."""


def repair_genome(genome: np.ndarray) -> np.ndarray:
    """Clamp a genome into the feasible box.

    Called after every crossover and mutation so the population is feasible by
    invariant rather than by inspection. Clipping rather than rejecting keeps
    selection pressure on the boundary, where good strategies often sit -- a
    reject-and-resample scheme wastes evaluations and biases the search away
    from the bounds.
    """
    return np.clip(np.asarray(genome, dtype=np.float64), LOW, HIGH)


def validate_sub_ledger(sub_ledger: pd.DataFrame, episode: Episode) -> pd.DataFrame:
    """Assert the decoded strategy is physically possible.

    Checks, in order of how badly each would corrupt the results:

    1. **Amounts positive and above the plausible floor.** A zero or negative
       transfer is meaningless; a $3 transfer is not laundering.
    2. **No self-transfers.** An account paying itself moves no money but would
       still generate features.
    3. **Transaction count within the cap**, so one candidate cannot dominate a
       generation's evaluation budget.
    4. **Per-account timestamps non-decreasing**, and in particular no account
       spending money before it received it. This is the one that would quietly
       invalidate the science: a strategy that sends from a mule before funding
       it is exploiting a simulation bug, not discovering an evasion.
    """
    if sub_ledger.empty:
        return sub_ledger

    if (sub_ledger["amount_paid"] < MIN_TRANCHE - 1e-6).any():
        raise ImplausibleStrategy(f"tranche below the ${MIN_TRANCHE:.0f} plausibility floor")
    if (sub_ledger["amount_received"] <= 0).any():
        raise ImplausibleStrategy("non-positive amount received")
    if (sub_ledger["from_account"] == sub_ledger["to_account"]).any():
        raise ImplausibleStrategy("self-transfer")
    if len(sub_ledger) > MAX_TRANSACTIONS:
        raise ImplausibleStrategy(f"{len(sub_ledger)} transactions exceeds the cap of {MAX_TRANSACTIONS}")

    hours = to_hours(sub_ledger["timestamp"], episode.epoch)
    if (hours < episode.start_hour - 1e-6).any():
        raise ImplausibleStrategy("transaction before the episode start")

    # No account may send twice at the same instant.
    #
    # This check exists because the GA exploited its absence. With zero delay it
    # emitted twelve $20k transfers from one account at one identical timestamp;
    # since the as-of feature queries exclude ties, none of those transactions
    # could see the others and every burst feature read zero, giving the
    # strategy a perfect 0.000 detection rate. It was exploiting the simulation,
    # not evading the detector -- a real bank sees twelve simultaneous
    # transfers. See decoder.MIN_GAP_HOURS.
    per_sender = pd.DataFrame({"sender": sub_ledger["from_account"].to_numpy(), "h": hours})
    gaps = per_sender.sort_values("h").groupby("sender", observed=True)["h"].diff().dropna()
    if len(gaps) and float(gaps.min()) < MIN_GAP_HOURS - 1e-9:
        raise ImplausibleStrategy(
            f"two transfers from one account only {float(gaps.min()) * 3600:.3f}s apart "
            f"(minimum {MIN_GAP_HOURS * 3600:.0f}s) -- simultaneous transfers are invisible "
            "to the as-of burst features and would not be possible in a real payment system"
        )

    # Causality: at no point may an account have sent more than it holds.
    # Mules start empty, so their outflow must always trail their inflow. The
    # source starts with the goal amount.
    order = np.argsort(hours, kind="stable")
    balance: dict[str, float] = {episode.src_account: float(episode.amount_goal)}
    senders = sub_ledger["from_account"].to_numpy(dtype=object)
    receivers = sub_ledger["to_account"].to_numpy(dtype=object)
    paid = sub_ledger["amount_paid"].to_numpy(dtype=float)
    received = sub_ledger["amount_received"].to_numpy(dtype=float)

    for i in order:
        s = senders[i]
        held = balance.get(s, 0.0)
        # Tolerance of a cent absorbs float noise in the fee arithmetic.
        if held + 0.01 < paid[i]:
            raise ImplausibleStrategy(
                f"account {s} sent {paid[i]:.2f} holding only {held:.2f} -- "
                "money spent before it arrived"
            )
        balance[s] = held - paid[i]
        balance[receivers[i]] = balance.get(receivers[i], 0.0) + received[i]

    return sub_ledger


def conservation_error(sub_ledger: pd.DataFrame, episode: Episode) -> float:
    """Absolute money-conservation error, in currency units.

    Should be ~0 for every genome (invariant 2). Sums to: what left the source,
    plus what the source started with, minus what every account now holds,
    minus fees. Used by ``tests/test_adversary.py`` across thousands of random
    genomes -- the point is that it holds for arbitrary crossover offspring, not
    just for genomes the GA happens to produce.
    """
    if sub_ledger.empty:
        return 0.0

    inflow: dict[str, float] = {}
    outflow: dict[str, float] = {}
    for s, r, p, rc in zip(
        sub_ledger["from_account"],
        sub_ledger["to_account"],
        sub_ledger["amount_paid"],
        sub_ledger["amount_received"],
    ):
        outflow[s] = outflow.get(s, 0.0) + float(p)
        inflow[r] = inflow.get(r, 0.0) + float(rc)

    fees = float((sub_ledger["amount_paid"] - sub_ledger["amount_received"]).sum())
    held = {a: inflow.get(a, 0.0) - outflow.get(a, 0.0) for a in set(inflow) | set(outflow)}
    # Everything the source put in must now be held somewhere or paid in fees.
    injected = outflow.get(episode.src_account, 0.0) - inflow.get(episode.src_account, 0.0)
    settled = sum(v for a, v in held.items() if a != episode.src_account)
    return abs(injected - settled - fees)
