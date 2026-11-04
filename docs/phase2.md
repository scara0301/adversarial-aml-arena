# Phase 2 — The genetic-algorithm adversary

*Objective 2: a GA adversary whose genome encodes realistic evasion actions — structuring, timing
delays, intermediate-account routing — constrained to plausible transaction behaviour.*

*Objective 3: a fitness function balancing detection evasion against completion of the
adversary's underlying laundering objective, evolved across generations.*

Status: **complete**. Reproduce with:

```bash
uv run python scripts/evolve_adversary.py
```

## Results

Population 100, 40 generations, 5 episodes per evaluation — 20,500 candidate evaluations in about
**6 minutes**. All strategies re-scored afterwards on 20 freshly drawn episodes, so the comparison
is like-for-like rather than each strategy's own lucky draw.

> **These numbers are from a re-run.** The first run (recall 0.767 → 0.247, gap 0.520) used a
> decoder that stamped every adversary transaction with a constant fee, which Phase 3 showed is a
> fingerprint (see [phase3.md](phase3.md)). The decoder was fixed and this phase re-run. The
> headline moved by less than 0.02 — the round-0 detector barely used the affected feature — so the
> conclusion stands. The *winning genome* did change, and some details I first reported about it
> did not replicate; see "What the adversary learned".

### The headline: detector recall against each strategy

| Strategy | Detector recall | Fitness | Completion | Evasion |
|---|---|---|---|---|
| naive (hand-written) | **0.750** | 0.260 | 0.999 | 0.261 |
| random (unevolved, mean of 10) | 0.748 | 0.002 | — | — |
| **evolved (GA best)** | **0.231** | 0.771 | 0.999 | 0.773 |

**Recall gap: 0.519.** The detector catches three quarters of naive laundering and under a quarter
of the evolved strategy, while the evolved strategy still delivers 99.9% of the goal amount within
the deadline. That gap is Phase 2's deliverable: the proof there is something for the arms race to
race against, and the starting point of Phase 3.

### Convergence

| Generation | Best fitness | Mean fitness | Evasion | Diversity |
|---|---|---|---|---|
| 0 | 0.318 | 0.029 | 0.326 | 0.392 |
| 10 | 0.906 | 0.605 | 0.917 | 0.346 |
| 20 | 0.906 | 0.680 | 0.917 | 0.312 |
| 40 | 0.911 | 0.707 | 0.913 | 0.358 |

The **population mean** is the signal that selection works (0.029 → 0.707, a 24× rise).
Best-of-generation matters less: elitism carries a lucky initial draw forward untouched, so a
flat best curve would not mean the GA failed to learn. Diversity stays near 0.35 — fitness sharing
is keeping niches alive, which is what gives Phase 3 a varied strategy pool rather than eight
copies of one winner.

Artifacts: `reports/phase2_metrics.json`, `reports/phase2_convergence.png`,
`reports/phase2_recall_comparison.png`, `reports/phase2_best_strategy_example.csv`.

## What the adversary learned

The winning genome:

| Gene | Naive | Evolved |
|---|---|---|
| typology_template | stack | stack |
| n_hops | 1 | 1 |
| n_splits | 3 | 8 |
| split_skew | 0.5 | 0.022 |
| **threshold_hug** | 0.0 | **0.675** |
| mean_delay_h | 0.5 | 0.0 |
| delay_jitter | 0.1 | 0.126 |
| business_hours_bias | 0.0 | 0.171 |
| mule_pool_size | 1 | 6 |
| mule_reuse | 1.0 | 0.588 |
| format_mix | 0.5 | 0.039 |
| dormancy_h | 0.0 | 26.0 |

Decoded, that is 13 transfers of \$13k–\$26k from one account to another, one minute apart: eight
in an evening burst, five the next morning, summing to exactly \$250,000.00.

**The robust finding: the GA did not fully structure.** `threshold_hug` evolved to 0.675, not 1.0 —
a tranche cap of about \$28,000, comfortably *above* the \$10,000 reporting threshold. It reached
0.678 in the first run and 0.675 in this one, from independent runs of the same search: this is the
one number in the genome that is reproducibly *learned* and not drift. Every transaction in the
winning strategy is over the threshold that the detector's strongest per-transaction feature is
built around.

That looks wrong until you consider what full structuring costs. Driving `threshold_hug` to 1.0
would require 26+ tranches instead of 13, and those extra transactions create exactly the burst
the relational features (`out_cnt_1h`, `out_cnt_24h`, `burst_ratio`, `gather_scatter_24h`) are
designed to catch. Against a graph-aware detector, structuring below the reporting threshold looks
counterproductive: it trades a modest amount-feature gain for a large burst-feature penalty. The
winning move is to sit in the *fat tail of the benign amount distribution* — \$13k–\$26k transfers
are unremarkable in traffic whose 95th percentile is around \$24k — and keep the transaction count
low.

**What I over-read the first time.** The first write-up of this phase also reported that
`business_hours_bias` "rose from 0 to 0.890, hiding the activity inside the benign diurnal rhythm",
that `format_mix` "went to 0.000, pinning every transfer to the lowest-scrutiny formats", and that
the strategy used 6 tranches. In this run those genes are 0.171, 0.039 and 8. They did not
replicate, so they were drift and not discoveries. The reason is visible in the genome: with
`n_hops = 1` there is a single layer of routing, so `mule_pool_size`, `mule_reuse` and
`dormancy_h` have nothing to act on, and a gene with no fitness consequence wanders. Do not read a
value into a gene the winning strategy does not use. (The decoded strategy is also nearly identical
across the two runs — same shape, same size, same two bursts — which is the better evidence that
the *behaviour* is stable even where the *genome* is not.)

## Design decisions

### Money conservation is a property of the representation (invariant 2)

No gene holds an absolute amount. Every value is a count, a fraction or a shape parameter, and
the laundering goal lives in the `Episode` rather than in the genome. The decoder always
allocates exactly the balance it is handed: tranches are `balance * normalised_weights`, and the
final tranche is set to `balance - sum(others)` to absorb the last float epsilon.

The consequence is that **arbitrary crossover offspring conserve money by construction**. A design
storing tranche amounts in the genome would have to repair or penalise every offspring, and would
leak money constantly. Measured over 600 random crossover offspring: maximum conservation error
**\$0.0000000002** on a \$250,000 goal (≈4e-15 relative), zero implausible strategies.

### `threshold_hug` caps the largest tranche, not the mean

Two iterations were needed here, and both were caught by tests rather than by inspection.

The first version interpolated linearly, `T * (0.5 + 0.49 * hug)`, which bottomed out at a
\$5,000 target — so *every* genome produced 50 sub-threshold tranches for a \$250k goal. The gene
had almost no range, and worse, the "naive" baseline was already evading the threshold feature,
which would have made the headline comparison meaningless. Replaced with a geometric
interpolation from "one transfer of the whole balance" to "just under \$10,000".

The second version sized the tranche count from the *mean*, which let a skewed `split_skew` draw
hand one tranche a large share: at `threshold_hug = 1.0` it produced a \$31,756 tranche among 26
nominally compliant ones. A real structurer does not let one tranche blow the limit. The decoder
now draws, checks the maximum, and splits more finely if it overshoots — retrying costs nothing in
conservation terms because each draw independently sums to the balance, whereas *clipping* the
overshoot would have destroyed money and broken invariant 2.

### Hard repair, not penalty terms

`repair_genome` clips every genome into its bounds after each crossover and mutation, so the whole
population is always feasible. With penalty terms, fitness would measure a blend of evasion skill
and constraint violation — a score of 0.4 could mean a brilliant evader that broke one rule or a
mediocre one that broke none — and the GA would spend its search budget learning to be feasible
instead of learning to evade.

### The fitness function and its guards

```
evasion    = 1 - sum(amount_i * flagged_i) / sum(amount_i)
completion = delivered_to_destination_by_deadline / amount_goal
cost       = fees / amount_goal + 0.3 * elapsed / deadline
fitness    = completion * evasion - 0.1 * cost,   and 0 if completion < 0.95
```

A genetic algorithm is an adversarial search over your own scoring function, and it will find
every way to score well without doing the task. Each property below closes a specific loophole:

| Degenerate strategy | Why it would win | Guard |
|---|---|---|
| Move \$0 → never flagged | additive fitness rewards evasion alone | multiplicative form + hard `completion < 0.95 ⇒ 0` |
| One giant transfer | fewest transactions = fewest flag chances | amount-weighted evasion *(see correction below)* |
| Shred into \$1 dust | each tranche individually unremarkable | \$50 tranche floor, 120-transaction cap, per-transaction fee |
| Stall past the deadline | delay defeats burst features | post-deadline value does not count toward delivery |
| Hide behind one freak account | overfits a dataset artefact | fitness averaged over 5 independent episodes |
| Population collapses to clones | nothing varied left for Phase 3 | fitness sharing on genome distance |

The episode-averaging guard is the most expensive and the most important. Without it the GA finds
the one background account with thousands of counterparties where anything hides — a property of
this dataset, not an evasion strategy. It also multiplies evaluation cost by five, which is why
the copy-on-write overlay in `AccountGraph` matters so much.

## The exploit the GA found, and the fix

**The first completed run reported a detector recall of exactly 0.000 — total evasion.** That is
not a result to celebrate, and inspecting the decoded transactions showed why:

```
2026-03-15 16:53:54.272567770  ACC0013041 -> ACC0002720  $18,673.95
2026-03-15 16:53:54.272567770  ACC0013041 -> ACC0002720  $16,473.16
2026-03-15 16:53:54.272567770  ACC0013041 -> ACC0002720  $16,699.45
...   12 transactions, 1 distinct timestamp, time span 0.0 seconds
```

With `mean_delay_h` at 0, the decoder emitted every tranche at one identical timestamp. The as-of
window queries use `searchsorted(..., side="left")`, which excludes ties — so **none of those
twelve transactions could see any of the others**. Every burst feature read zero and the detector
was structurally blind to a burst that a real bank could not possibly miss.

This was the adversary exploiting the gap between the simulation and reality, not evading a
detector. It is exactly the class of degenerate optimum the fitness guards are meant to catch, and
it got through because the plan specified "strictly increasing per-account timestamps" as a
plausibility constraint and the decoder never enforced it.

**Fix:** a `MIN_GAP_HOURS` of one minute between consecutive transfers from the same account,
enforced in the decoder and asserted in `validate_sub_ledger`. Real payment systems timestamp to
the second and cannot process two transfers from one account at the same instant, so this is a
plausibility constraint rather than a modelling convenience. The separation is also re-applied
*after* the business-hours snap, because two times that both fall before 09:00 snap to exactly
09:00 and would reintroduce the collision.

`tests/test_adversary.py::test_no_account_sends_twice_at_the_same_instant` guards it permanently,
testing both the extreme case (all timing genes zeroed) and 200 random genomes.

After the fix, evasion fell from a perfect 1.000 to 0.906 and detector recall rose from 0.000 to
0.247 in that run (0.231 after the later decoder fix) — the honest numbers reported above.

The general lesson is worth recording for Phases 3–5: **when an adversarial search reports a
perfect score, the first hypothesis should be that it found a bug in the simulation.** Every
guard in `fitness.py` was written in anticipation of a specific loophole; this one was found by
the GA first, which is the failure mode to expect again.

## Correction, found in Phase 3: the "one giant transfer" guard did not guard

The guard table above lists amount-weighted evasion as the defence against the strategy "send
everything as one transfer", on the reasoning that flagging a single transfer would cost the
adversary 100% of its evasion score. That reasoning is only half a guard. Amount-weighting makes
*being flagged* expensive; it does nothing if the detector **does not flag** the transfer.

Phase 2 never found out because its GA happened not to reach a one-tranche genome in 40
generations. Phase 3's did, in round 1, and scored detector recall 0.000: the detector was blind to
a lone round-number transfer (see the "fourth artefact" section of [phase1.md](phase1.md) and
[phase3.md](phase3.md)). The fitness function has no defence against a degenerate strategy the
*detector* cannot see, and was never going to: that is a property of the detector, and is exactly
what the closed loop exists to find and close.

Nothing here changes the Phase 2 headline (0.750 → 0.231 in the re-run; 0.767 → 0.247 originally), which was measured as reported. But the
claim "this strategy is guarded against" was stronger than the evidence, and is withdrawn.

## Limitations

1. **The adversary is evolved against one fixed detector.** That is what Phase 2 asks for, but it
   makes these numbers an upper bound on evasion. A detector retrained on evolved strategies will
   be considerably harder to fool — the entire point of Phase 3.
2. **The evolved strategy uses one hop.** With `n_hops = 1` the routing genes (`mule_pool_size`,
   `mule_reuse`, `dormancy_h`) have little effect, so the multi-hop routing machinery is built and
   tested but not yet load-bearing in the winning strategy. That is a finding, not a defect: against
   *this* detector, extra hops cost fees and create more transactions without buying enough
   concealment. Phase 3's retrained detectors should change that calculus.
3. **The 120-transaction cap binds the search.** Strategies wanting very fine structuring cannot
   express it. The cap is deliberate — both for plausibility and for evaluation speed — but it is a
   boundary the GA sits near.
4. **Synthetic background.** As in Phase 1, these numbers are a pipeline result rather than a claim
   about real laundering.

## What Phase 3 needs from this

- `reports/phase2_metrics.json` → `diverse_strategies`: eight mutually distinct genomes from the
  final population, selected by `distinct_top` with a minimum genome separation. Taking the top
  eight by fitness alone would return eight near-identical genomes, and retraining on eight copies
  of one strategy teaches the detector almost nothing.
- The naive-vs-evolved recall gap (0.750 → 0.231) as round 0 of the arms-race curve.
- `AccountGraph.overlay` and the 4.8 ms-per-candidate evaluation cost, which is what makes
  re-evolving the adversary once per round affordable.
