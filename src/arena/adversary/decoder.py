"""Genome to transaction sub-ledger.

**Invariant 2 lives here.** The decoder is the only thing that turns relative
gene values into money, and it always allocates exactly the balance it was
given. Two mechanisms make conservation exact rather than approximate:

1. Tranche amounts are always ``balance * normalised_weights``. The weights come
   from a Dirichlet draw and are normalised to sum to 1, so the tranches sum to
   the balance by construction -- for any genome, including any crossover
   offspring.
2. The final tranche is set to ``balance - sum(others)`` rather than to its own
   weighted share, which removes the last few float-epsilon of drift. Without
   it, conservation holds to ~1e-12 relative; with it, it holds to the last
   representable bit.

## Why ``threshold_hug`` changes the tranche *count*, not the amounts

The obvious implementation of "hug the reporting threshold" is to compute
tranches and then pull the large ones down under $10,000. That silently destroys
money and breaks invariant 2. So instead ``threshold_hug`` sets a *target tranche
size*, from which the decoder derives how many tranches are needed:

    target = $10,000 * (0.5 + 0.49 * threshold_hug)
    k      = max(n_splits, ceil(balance / target))

The genome therefore expresses a preference about tranche size, the decoder
satisfies it by splitting more finely, and the amounts are never post-edited. If
a tranche still lands above the threshold the decoder leaves it there -- the
detector will very likely flag it, and the GA is free to discover that raising
``threshold_hug`` fixes the problem. Clipping it would hide exactly the pressure
the arms race is supposed to apply.

## The episode carries the goal, the genome does not

An :class:`Episode` fixes what the adversary is trying to achieve: move
``amount_goal`` from ``src_account`` to ``dst_account`` by ``deadline_hours``,
using mules that already exist in the background ledger. Keeping the goal out of
the genome is what lets the same evolved strategy be evaluated against several
independent episodes, which is the guard against overfitting one lucky corner of
the background graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from arena import schema
from arena.adversary.genome import TEMPLATES, decode_genes
from arena.features.account_graph import to_hours
from arena.data.synthetic import CURRENCY_P, fx_legs
from arena.schema import CURRENCIES, PAYMENT_FORMATS, REPORTING_THRESHOLD

#: Extra proportional fee charged at each hop, on top of the FX spread. Zero by
#: default, and it should stay zero.
#:
#: This used to be 0.002, and that was a fingerprint. Every decoded transaction
#: then carried ``amount_received = amount_paid * 0.998`` and a USD-to-USD
#: currency pair, so ``fx_loss`` was exactly 0.002 on every adversary row, while
#: benign traffic has 0 (93% of rows) or a spread between 0.5% and 3%. A detector
#: retrained on decoded rows could then learn "fx_loss == 0.002" -- which says
#: nothing about evasion. It did: measured on the first closed-loop run, a quarter
#: of the retrained detector's apparent catch rate was that fingerprint alone, and
#: it leaned on ``fx_loss`` ten times harder than the round-0 detector had.
#:
#: The cost of moving money through mules is now realised by the same FX process
#: benign and background-laundering rows use (``arena.data.synthetic.fx_legs``),
#: so an adversary row's currency fields are drawn from the same distribution as
#: everyone else's. A non-zero fee here reintroduces a constant nobody else has.
DEFAULT_FEE_RATE = 0.0

#: Smallest plausible tranche. Nobody launders $3, and without a floor the GA
#: shreds the goal into thousands of trivial transfers that no single detector
#: feature can see.
MIN_TRANCHE = 50.0

#: Hard cap on transactions per candidate strategy. A 120-transfer operation is
#: already large by real standards, and the cap keeps a single fitness
#: evaluation fast enough for a 20,000-evaluation GA run.
MAX_TRANSACTIONS = 120

#: Minimum gap between consecutive transactions from the same account, in hours
#: (one minute).
#:
#: This closes a real exploit the GA found. With ``mean_delay_h`` at 0 the decoder
#: emitted every tranche at one identical timestamp -- 12 transfers of ~$20k from
#: one account to one account, all at 16:53:54.272567770. Because the as-of
#: window queries use ``side="left"`` and therefore exclude ties, none of those
#: transactions could see any of the others: every burst feature read zero and
#: the detector's recall on the strategy was exactly 0.000.
#:
#: That is not an evasion strategy, it is the adversary exploiting a gap between
#: the simulation and reality. A real bank sees twelve simultaneous $20k
#: transfers. Payment systems timestamp to the second and cannot process two
#: transfers from one account at literally the same instant, so a minimum
#: separation is a plausibility constraint, not a modelling convenience -- and
#: the plan called for strictly increasing per-account timestamps from the start.
MIN_GAP_HOURS = 1.0 / 60.0

#: Formats ordered by scrutiny, matching features.pipeline._FORMAT_SCRUTINY.
#: ``format_mix`` slides along this list.
_FORMATS_BY_SCRUTINY = ("ACH", "Cheque", "Credit Card", "Reinvestment", "Wire", "Cash", "Bitcoin")


@dataclass(slots=True)
class Episode:
    """One laundering task: move money from A to B by a deadline.

    ``mule_pool`` holds accounts drawn from the *background* ledger. Restricting
    mules to accounts that already exist is a plausibility constraint that
    matters: an adversary permitted to invent fresh accounts would trivially
    evade every account-history feature, which is not an evasion strategy but a
    hole in the simulation.
    """

    src_account: str
    dst_account: str
    amount_goal: float
    deadline_hours: float
    start_hour: float
    mule_pool: list[str]
    bank_of: dict[str, int] = field(default_factory=dict)
    epoch: pd.Timestamp = field(default_factory=lambda: pd.Timestamp("2026-01-01"))

    def bank(self, account: str) -> int:
        return int(self.bank_of.get(account, 0))


def _tranche_amounts(
    balance: float, n_splits: int, split_skew: float, threshold_hug: float, rng: np.random.Generator
) -> np.ndarray:
    """Split ``balance`` into tranches summing to exactly ``balance``.

    See the module docstring: ``threshold_hug`` raises the tranche *count* so
    the amounts land near the target size, and the amounts themselves are never
    adjusted afterwards.
    """
    # Target tranche size, interpolated geometrically between "do not structure
    # at all" (one transfer of the whole balance) and "sit just under the
    # reporting threshold".
    #
    # An earlier linear version, T * (0.5 + 0.49 * hug), bottomed out at $5,000 --
    # so even threshold_hug = 0 produced 50 sub-threshold tranches for a $250k
    # goal. Every genome structured automatically, which meant the gene had
    # almost no range to act over and, worse, the "naive" baseline was already
    # evading the threshold feature. The GA could not discover structuring
    # because structuring was mandatory.
    #
    # Geometric rather than linear interpolation because the endpoints differ by
    # orders of magnitude; a linear blend would spend most of its range on
    # implausibly large tranches.
    hug_target = REPORTING_THRESHOLD * 0.99
    balance = max(balance, 1.0)
    target = (balance ** (1.0 - threshold_hug)) * (hug_target**threshold_hug)

    # ``target`` caps the *largest* tranche, not the average. Sizing k from the
    # mean is not enough: a skewed Dirichlet draw (split_skew near 1) hands one
    # tranche a large share, and at threshold_hug = 1.0 that produced a $31,756
    # tranche among 26 nominally sub-threshold ones -- three times over the
    # reporting limit, which is not what "hug the threshold" can be allowed to
    # mean. A real structurer does not let one tranche blow the limit.
    #
    # So: draw, and if the largest tranche overshoots, split more finely and
    # draw again. Each draw is independent and sums to the balance exactly, so
    # retrying costs nothing in conservation terms. Clipping the overshoot
    # instead would have destroyed money and broken invariant 2.
    k = max(int(n_splits), int(np.ceil(balance / max(target, 1.0))))
    tranches = np.array([balance])

    for _ in range(8):
        # Never split below the plausible minimum, and never exceed the cap.
        k = max(1, min(k, MAX_TRANSACTIONS, max(1, int(balance // MIN_TRANCHE))))

        # High concentration gives near-equal tranches; low gives a skewed spread.
        concentration = 30.0 * (1.0 - split_skew) + 0.4
        weights = rng.dirichlet(np.full(k, concentration))
        weights = weights / weights.sum()

        # Give every tranche the plausibility floor first, then distribute only
        # the remainder by weight. A plain ``balance * weights`` lets a highly
        # skewed draw produce dust tranches of a few dollars, which the
        # plausibility check rightly rejects. Reserving the floor up front makes
        # the minimum structural rather than something to validate and retry,
        # and it keeps conservation exact because the reserved floors plus the
        # weighted remainder sum to the balance by construction.
        remainder = balance - k * MIN_TRANCHE
        tranches = MIN_TRANCHE + remainder * weights

        # Exact conservation: absorb all float drift into the final tranche.
        if k > 1:
            tranches[-1] = balance - float(tranches[:-1].sum())
        else:
            tranches[0] = balance

        at_cap = k >= min(MAX_TRANSACTIONS, max(1, int(balance // MIN_TRANCHE)))
        if float(tranches.max()) <= target * 1.0001 or at_cap:
            break
        k = int(k * 1.4) + 1

    return tranches


def _route_layers(template: str, n_hops: int, src: str, dst: str, mules: list[str]) -> list[list[str]]:
    """Account layers the money passes through, from ``[src]`` to ``[dst]``.

    Each template produces the topology its name implies, so the adversary is
    executing the same typology vocabulary the detector was trained on. Its task
    is to execute one of these shapes *without being flagged*, not to invent a
    shape nobody has seen.
    """
    if not mules or n_hops <= 1:
        return [[src], [dst]]

    if template == "fan-out":
        # One wide middle layer: src sprays across many mules, which converge.
        width = max(1, min(len(mules), n_hops * 2))
        return [[src], mules[:width], [dst]]

    if template == "gather-scatter":
        # Wide collection layer, then a single hub, then out.
        width = max(1, min(len(mules) - 1, n_hops * 2)) if len(mules) > 1 else 1
        hub = [mules[-1]]
        return [[src], mules[:width], hub, [dst]]

    # "stack" and "cycle" both layer sequentially, one mule per hop. The cycle
    # template differs only in the extra return edge added by the caller.
    depth = max(1, min(n_hops - 1, len(mules)))
    return [[src], *[[mules[i]] for i in range(depth)], [dst]]


def _hour_of_day(t: float) -> float:
    return t % 24.0


def _snap_business_hours(t: float, bias: float, rng: np.random.Generator) -> float:
    """Push ``t`` forward into 09:00-17:00 with probability ``bias``.

    Forward only, never backward. Moving a transaction earlier could place it
    before the transfer that funded it, producing a sub-ledger that is not just
    implausible but causally impossible. Waiting for the bank to open is also
    what actually happens.
    """
    if rng.random() >= bias:
        return t
    hour = _hour_of_day(t)
    if 9.0 <= hour < 17.0:
        return t
    if hour < 9.0:
        return t + (9.0 - hour)
    return t + (24.0 - hour) + 9.0  # next morning


def _pick_format(format_mix: float, rng: np.random.Generator) -> str:
    """Payment format, biased along the scrutiny axis by ``format_mix``.

    A little randomness is kept so the adversary cannot collapse onto a single
    format -- which would itself be a detectable regularity.
    """
    n = len(_FORMATS_BY_SCRUTINY)
    centre = format_mix * (n - 1)
    idx = int(np.clip(round(rng.normal(centre, 0.8)), 0, n - 1))
    return _FORMATS_BY_SCRUTINY[idx]


def decode(
    genome: np.ndarray,
    episode: Episode,
    rng: np.random.Generator,
    fee_rate: float = DEFAULT_FEE_RATE,
) -> pd.DataFrame:
    """Turn a genome into a canonical-schema sub-ledger.

    Returns transactions tagged ``source="adversary"`` and ``is_laundering=1``,
    labelled with the typology its template implements. Money is conserved
    exactly at every hop, minus the explicit per-hop fee.

    The returned frame is what the fitness function hands to ``featurize`` --
    never a feature vector (invariant 1).
    """
    p = decode_genes(genome)
    template = TEMPLATES[int(p["typology_template"])]

    mules = [m for m in episode.mule_pool[: int(p["mule_pool_size"])]
             if m not in (episode.src_account, episode.dst_account)]
    layers = _route_layers(template, int(p["n_hops"]), episode.src_account, episode.dst_account, mules)

    mean_delay = float(p["mean_delay_h"])
    jitter = float(p["delay_jitter"])
    bias = float(p["business_hours_bias"])
    dormancy = float(p["dormancy_h"])
    reuse = float(p["mule_reuse"])

    records: list[dict] = []
    holdings: dict[str, float] = {episode.src_account: float(episode.amount_goal)}
    # Per-account clock, so timestamps are strictly increasing per account.
    clock: dict[str, float] = {episode.src_account: float(episode.start_hour)}

    for hop, receivers in enumerate(layers[1:]):
        next_holdings: dict[str, float] = {}

        for sender, balance in holdings.items():
            if balance < MIN_TRANCHE or len(records) >= MAX_TRANSACTIONS:
                continue

            tranches = _tranche_amounts(
                balance, int(p["n_splits"]), float(p["split_skew"]), float(p["threshold_hug"]), rng
            )
            # Currency and FX come from the process every other row uses -- see
            # DEFAULT_FEE_RATE. One batched draw per sender keeps this cheap.
            cur_idx = rng.choice(len(CURRENCIES), size=len(tranches), p=CURRENCY_P)
            pay_cur, recv_cur, recv_amt = fx_legs(rng, tranches, cur_idx)

            # Money sits at each hop before moving on: the dormancy gene.
            t = clock.get(sender, float(episode.start_hour)) + dormancy * (hop > 0)

            for j, amount in enumerate(tranches):
                if len(records) >= MAX_TRANSACTIONS:
                    break

                # mule_reuse at 1.0 funnels every tranche through one receiver;
                # at 0.0 it spreads them across the layer.
                if reuse >= 1.0 or len(receivers) == 1:
                    receiver = receivers[0]
                elif rng.random() < reuse:
                    receiver = receivers[0]
                else:
                    receiver = receivers[j % len(receivers)]
                if receiver == sender:
                    receiver = receivers[(j + 1) % len(receivers)]
                    if receiver == sender:
                        continue

                # Gamma-distributed gap: positive by construction, with the
                # requested mean and jitter as its coefficient of variation.
                if mean_delay > 0 and jitter > 0:
                    shape = max(1.0 / (jitter**2), 0.15)
                    gap = float(rng.gamma(shape, mean_delay / shape))
                else:
                    gap = mean_delay
                t = _snap_business_hours(t + max(gap, MIN_GAP_HOURS), bias, rng)

                # Enforce strict separation after snapping as well. Two times
                # that both fall before 09:00 on the same day snap to exactly
                # 09:00, so the snap itself can reintroduce the collision that
                # MIN_GAP_HOURS exists to prevent.
                previous = clock.get(sender)
                if previous is not None and t < previous + MIN_GAP_HOURS:
                    t = previous + MIN_GAP_HOURS

                received = float(recv_amt[j]) * (1.0 - fee_rate)
                fmt = _pick_format(float(p["format_mix"]), rng)

                records.append(
                    {
                        "timestamp": episode.epoch + pd.to_timedelta(t, unit="h"),
                        "from_bank": episode.bank(sender),
                        "from_account": sender,
                        "to_bank": episode.bank(receiver),
                        "to_account": receiver,
                        "amount_paid": float(amount),
                        "payment_currency": str(pay_cur[j]),
                        "amount_received": float(received),
                        "receiving_currency": str(recv_cur[j]),
                        "payment_format": fmt,
                        "is_laundering": 1,
                        "typology": template,
                        "pattern_id": 0,
                        "source": "adversary",
                    }
                )
                next_holdings[receiver] = next_holdings.get(receiver, 0.0) + received
                clock[sender] = t
                clock[receiver] = max(clock.get(receiver, 0.0), t)

        holdings = next_holdings
        if not holdings:
            break

    # The cycle template closes a loop back to the source, which is the
    # topological signature the detector's cycle features look for. A small
    # token amount: the point is the edge, not the value.
    #
    # The token is sent by the *destination*, not by an intermediate mule. A
    # mule has already forwarded everything it received and holds nothing, so a
    # return edge from a mule spends money that does not exist -- it failed the
    # causality check on 125 of 400 random genomes. The destination is the only
    # account holding a balance once the route completes, and "money completes
    # the loop back to its origin" is the more faithful reading of the cycle
    # typology anyway. delivered_amount() nets this outflow off completion, so
    # the accounting stays honest.
    if template == "cycle" and records and len(records) < MAX_TRANSACTIONS:
        sender = episode.dst_account
        token = max(MIN_TRANCHE, float(episode.amount_goal) * 0.001)
        t_cur_idx = rng.choice(len(CURRENCIES), size=1, p=CURRENCY_P)
        t_cur, t_recv_cur, t_recv = fx_legs(rng, np.array([token]), t_cur_idx)
        t = clock.get(sender, float(episode.start_hour)) + max(dormancy, 1.0)
        records.append(
            {
                "timestamp": episode.epoch + pd.to_timedelta(t, unit="h"),
                "from_bank": episode.bank(sender),
                "from_account": sender,
                "to_bank": episode.bank(episode.src_account),
                "to_account": episode.src_account,
                "amount_paid": token,
                "payment_currency": str(t_cur[0]),
                "amount_received": float(t_recv[0]) * (1.0 - fee_rate),
                "receiving_currency": str(t_recv_cur[0]),
                "payment_format": _pick_format(float(p["format_mix"]), rng),
                "is_laundering": 1,
                "typology": "cycle",
                "pattern_id": 0,
                "source": "adversary",
            }
        )

    if not records:
        return schema.empty_ledger()

    df = pd.DataFrame.from_records(records)
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return schema.coerce(df)


def delivered_amount(
    sub_ledger: pd.DataFrame, episode: Episode, fee_rate: float = DEFAULT_FEE_RATE
) -> float:
    """Value that actually reached the destination by the deadline.

    Net: inflows to the destination minus anything it sent back out, so a
    strategy cannot inflate completion by bouncing money through the target.
    Transfers after the deadline are excluded entirely -- that is what stops the
    GA from evading the detector's burst features by simply stalling forever.
    """
    if sub_ledger.empty:
        return 0.0

    hours = to_hours(sub_ledger["timestamp"], episode.epoch)
    in_time = hours <= episode.start_hour + episode.deadline_hours
    dst = episode.dst_account

    arrived = float(
        sub_ledger.loc[in_time & (sub_ledger["to_account"] == dst), "amount_received"].sum()
    )
    left = float(
        sub_ledger.loc[in_time & (sub_ledger["from_account"] == dst), "amount_paid"].sum()
    )
    return max(arrived - left, 0.0)


def total_fees(sub_ledger: pd.DataFrame) -> float:
    """Value lost to fees across the whole strategy."""
    if sub_ledger.empty:
        return 0.0
    return float((sub_ledger["amount_paid"] - sub_ledger["amount_received"]).sum())


def elapsed_hours(sub_ledger: pd.DataFrame, episode: Episode) -> float:
    """Wall-clock span of the strategy, from the episode start."""
    if sub_ledger.empty:
        return 0.0
    hours = to_hours(sub_ledger["timestamp"], episode.epoch)
    return float(max(hours.max() - episode.start_hour, 0.0))
