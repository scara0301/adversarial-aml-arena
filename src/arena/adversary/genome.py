"""The adversary's genome: 12 genes describing an evasion strategy.

Objective 2 asks for a genome encoding "realistic evasion actions --
structuring, timing delays, and intermediate-account routing -- constrained to
plausible transaction behaviour". These twelve genes cover exactly that:

* **structuring** -- ``n_splits``, ``split_skew``, ``threshold_hug``
* **timing** -- ``mean_delay_h``, ``delay_jitter``, ``business_hours_bias``,
  ``dormancy_h``
* **routing** -- ``typology_template``, ``n_hops``, ``mule_pool_size``,
  ``mule_reuse``
* **scrutiny** -- ``format_mix``

## The single most important property

**No gene holds an absolute amount.** Every quantity here is relative: a count,
a fraction, a shape parameter. The laundering goal (move X from source to
destination) lives in the :class:`~arena.adversary.decoder.Episode`, not in the
genome, and the decoder always allocates exactly X.

That is what makes invariant 2 hold. Money conservation is not a constraint the
GA has to be penalised into respecting -- it is a property of the representation.
Arbitrary crossover between any two genomes, and arbitrary mutation of any gene,
produces a strategy that still moves exactly the goal amount, because there is
no absolute quantity anywhere for an operator to corrupt. Naive designs that put
tranche amounts in the genome have to repair or penalise every offspring, and
they leak money constantly.

Fixed length, twelve genes: small enough for a GA to converge in tens of
generations, expressive enough to cover all three evasion axes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Gene:
    """One gene's name, bounds and type.

    ``is_int`` genes are rounded on decode. They stay float internally so the
    GA's real-valued crossover and Gaussian mutation work uniformly across the
    whole vector -- special-casing integer genes in the operators is a common
    source of subtle bias.
    """

    name: str
    low: float
    high: float
    is_int: bool = False
    note: str = ""


#: The genome specification. Order is the vector order and must not change
#: without regenerating saved genomes.
GENES: tuple[Gene, ...] = (
    Gene("typology_template", 0, 3, True, "0 fan-out, 1 cycle, 2 gather-scatter, 3 stack"),
    Gene("n_hops", 1, 4, True, "routing depth through intermediate accounts"),
    Gene("n_splits", 1, 12, True, "structuring: number of tranches per leg"),
    Gene("split_skew", 0.0, 1.0, False, "0 equal tranches, 1 highly uneven"),
    Gene("threshold_hug", 0.0, 1.0, False, "how close tranches sit under $10k"),
    Gene("mean_delay_h", 0.0, 72.0, False, "mean inter-transaction delay"),
    Gene("delay_jitter", 0.0, 1.0, False, "coefficient of variation of delays"),
    Gene("business_hours_bias", 0.0, 1.0, False, "probability of landing in 09:00-17:00"),
    Gene("mule_pool_size", 1, 10, True, "intermediate accounts available"),
    Gene("mule_reuse", 0.0, 1.0, False, "0 fresh mule per tranche, 1 reuse one"),
    Gene("format_mix", 0.0, 1.0, False, "0 low-scrutiny formats, 1 high-scrutiny"),
    Gene("dormancy_h", 0.0, 168.0, False, "hold time at each intermediate hop"),
)

N_GENES = len(GENES)
GENE_INDEX = {g.name: i for i, g in enumerate(GENES)}
LOW = np.array([g.low for g in GENES], dtype=np.float64)
HIGH = np.array([g.high for g in GENES], dtype=np.float64)

#: Routing shapes the ``typology_template`` gene selects. Deliberately the same
#: vocabulary the detector was trained to recognise -- the adversary's job is to
#: execute these shapes in a way that evades, not to invent unseen ones.
TEMPLATES = ("fan-out", "cycle", "gather-scatter", "stack")


def random_genome(rng: np.random.Generator) -> np.ndarray:
    """Uniform random genome inside the bounds."""
    return rng.uniform(LOW, HIGH)


def decode_genes(genome: np.ndarray) -> dict[str, float | int]:
    """Genome vector to a named, correctly-typed parameter dict.

    Clips to bounds and rounds integer genes, so a caller can never see an
    out-of-range or fractional value even if an operator produced one. This is
    the only place gene types are interpreted.
    """
    values = np.clip(np.asarray(genome, dtype=np.float64), LOW, HIGH)
    out: dict[str, float | int] = {}
    for i, gene in enumerate(GENES):
        out[gene.name] = int(round(values[i])) if gene.is_int else float(values[i])
    return out


def describe(genome: np.ndarray) -> str:
    """Readable strategy summary, for reports and the dashboard."""
    p = decode_genes(genome)
    return "\n".join(
        [
            f"  template            {TEMPLATES[int(p['typology_template'])]}",
            f"  routing hops        {p['n_hops']}",
            f"  tranches per leg    {p['n_splits']}",
            f"  tranche skew        {p['split_skew']:.3f}",
            f"  threshold hug       {p['threshold_hug']:.3f}",
            f"  mean delay          {p['mean_delay_h']:.1f} h",
            f"  delay jitter        {p['delay_jitter']:.3f}",
            f"  business-hours bias {p['business_hours_bias']:.3f}",
            f"  mule pool           {p['mule_pool_size']}",
            f"  mule reuse          {p['mule_reuse']:.3f}",
            f"  format mix          {p['format_mix']:.3f}",
            f"  hop dormancy        {p['dormancy_h']:.1f} h",
        ]
    )


def to_dict(genome: np.ndarray) -> dict:
    """JSON-serialisable form, for metrics artifacts."""
    p = decode_genes(genome)
    p["typology_template_name"] = TEMPLATES[int(p["typology_template"])]
    return p


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised Euclidean distance between two genomes.

    Used for the diversity guard: the GA converges to clones if left alone, and
    a population of clones gives Phase 3 nothing varied to feed back into the
    detector's training set. Normalising by each gene's range stops
    ``dormancy_h`` (range 168) from dominating ``split_skew`` (range 1).
    """
    span = np.maximum(HIGH - LOW, 1e-9)
    return float(np.linalg.norm((np.asarray(a) - np.asarray(b)) / span) / np.sqrt(N_GENES))
