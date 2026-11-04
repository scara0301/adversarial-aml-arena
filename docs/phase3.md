# Phase 3 — The closed loop

*Objective 4: fold the adversary's evasive strategies back into the detector's training data, retrain
over successive rounds, and measure detector recall against adversary success across them.*

Status: **complete, with caveats that matter** (see "What this does and does not show"). Reproduce:

```bash
uv run python scripts/run_arena.py                    # the 5 rounds, ~30 min
uv run python scripts/reevaluate_arena.py --episodes 200   # sharper measurement, ~3 min
```

## Result

Five rounds. `D_r` is the detector at round `r` (`D_0` is the Phase 1 detector); `A_r` is the
adversary evolved against it. Every adversary number is measured on **200 fixed episodes from a later
period of the ledger than anything either side trained on**, with a 95% bootstrap interval over
episodes. Clean-holdout numbers are on the frozen holdout that never sees adversary data.

| Round | Detector recall vs. its contemporary adversary `A_r` | Adversary success rate | Holdout recall @ 0.5% | Holdout PR-AUC |
|---|---|---|---|---|
| 0 | **0.125** [0.11–0.14] | 0.99 | 0.538 | 0.454 |
| 1 | 0.275 [0.21–0.34] | 0.72 | 0.519 | 0.437 |
| 2 | 0.295 [0.23–0.36] | 0.70 | 0.529 | 0.441 |
| 3 | 0.365 [0.30–0.43] | 0.64 | 0.516 | 0.427 |
| 4 | **0.510** [0.48–0.54] | 0.56 | 0.523 | 0.430 |
| 5 (final `D_5`) | — | — | 0.509 | 0.421 |

*Adversary success* = the money arrived by the deadline and more than half its value escaped the alert
queue, per episode.

**The detector gains ground on its contemporary adversary every round and never gets ahead of it.**
Detector recall against the adversary built to beat it rises from 0.13 to 0.51, and the adversary's
success rate falls from 0.99 to 0.56 — but the adversary still wins a majority of episodes after four
rounds of feedback. There is no collapse on clean data, but there is a small cost: holdout recall at the alert budget falls
from 0.538 to 0.509 (−0.029) and PR-AUC from 0.454 to 0.421 (−0.033). With 1,132 holdout positives the
standard error on recall is about 0.015, so no single round's dip means much, but the PR-AUC drift is
fairly steady and I read the pair as a real, modest price for robustness.

### The cross-play matrix

Every detector (rows) against every adversary's champion (columns), detector recall; bold is the
diagonal. This is the result a single curve cannot give you, because it separates a detector that is
getting harder to fool from one that is merely chasing the strategy it was last shown.

| | A_0 | A_1 | A_2 | A_3 | A_4 |
|---|---|---|---|---|---|
| **D_0** | **0.13** | 0.20 | 0.17 | 0.20 | 0.19 |
| **D_1** | 0.67 | **0.28** | 0.27 | 0.21 | 0.47 |
| **D_2** | 0.64 | 0.35 | **0.29** | 0.28 | 0.43 |
| **D_3** | 0.58 | 0.41 | 0.39 | **0.36** | 0.49 |
| **D_4** | 0.57 | 0.50 | 0.55 | 0.53 | **0.51** |
| **D_5** | 0.54 | 0.47 | 0.45 | 0.46 | 0.52 |

Read by region:

- **The lesson** (the cell just below each diagonal: `D_(r+1)` vs `A_r`). Feedback works hardest the
  first time: `D_1` catches `A_0` at 0.67 against `D_0`'s 0.13, a gain of **+0.54**. After that the
  per-round gains are modest: +0.07, +0.09, +0.17, +0.01.
- **Memory** (older adversaries against newer detectors). Retained, but not uniformly. Comparing the
  recall right after a strategy was shown with the final detector's: `A_0` falls 0.67 → 0.54 and `A_3`
  0.53 → 0.46, while `A_1` *rises* 0.35 → 0.47 and `A_2` 0.39 → 0.45 (and `A_4` is flat at 0.52). The
  rises are plausibly later strategies resembling earlier ones, so feedback on one helps the other;
  the falls fit the cap being split into equal slices, so each round's slice shrinks as rounds
  accumulate. There is no consistent forgetting trend, and with one seed I would not claim one.
- **Progress** (newer adversaries against older detectors). There is **none**. Against `D_0`, the
  later adversaries are caught at 0.20, 0.17, 0.20, 0.19 — *more* than `A_0`'s 0.13. The adversary is
  not climbing a ladder; each `A_r` is a response to its own detector, not an escalation of the last.
- **The end state.** `D_5` catches every generation of adversary at roughly 0.45–0.54: about half.

### What the adversary does

It alternates between two families rather than escalating.

| | Strategy | Transactions per episode |
|---|---|---|
| `A_0`, `A_4` | multi-hop fan-out, ~3 hops, long dormancy (143 h and 168 h) | ~11 |
| `A_1`, `A_2`, `A_3` | **one large transfer** (1 hop, 1 tranche, no structuring) | 1.0 |

The single large transfer keeps working (64–72% success) even after the detector has trained on it for
rounds. The reason is in the data, not in the adversary: the synthetic *training period* contains **no
laundering above $146k** (max $145,883) against 702 benign transfers above $250k, so a detector has
almost nothing to generalise "a large transfer can be laundering" from. (Over the whole ledger the
figures are $148,459 and 1,223.) See limitation 3 below.

### Generalisation probes

Two strategies the detector was never trained on are scored every round, to separate "learned the
strategies it was shown" from "learned something general". Recall against the fixed naive strategy
rises from 0.49 to 0.69, and against ten unseen random strategies from 0.60 to 0.68. Both improve,
modestly. Most of the gain in the matrix is specific to the strategies that were shown.

## What went wrong on the way

Phase 3 took three full runs, and each one found a different flaw in *my own experiment*. They are
recorded in order because each is a more specific instance of the same lesson in `CLAUDE.md`: **a
suspiciously good adversarial score is a bug until the decoded transactions say otherwise**.

### Run 1 — a round-number loophole (detector recall 0.000)

Round 1's adversary scored **detector recall 0.000, evasion 1.000**. Reading its decoded
transactions: one $250,000.00 cheque. It was not a simulation bug — the transaction was valid, conserved
money and respected causality — but the detector scored it 0.0023 against a threshold of 0.47.

The first hypothesis, "the detector is blind to large amounts", was half right. The data does have no
laundering above $146k. But a test showed something sharper: `D_1` flags a **non-round**
$249,117.43 (score 0.83) and not exactly **$250,000.00** (score 0.008). Every amount in my sweep was
round, and every one scored about zero. The blind spot was *roundness*:

- 21.6% of benign amounts are round multiples of $100 — **0.00% of laundering amounts are**. "Round"
  therefore means "benign" with certainty, a free tell. `amount_mod_100` was the #2 feature in Phase 1
  all along.
- With one tranche the decoder sends exactly the episode goal, and **every episode had the same, round,
  goal**. A one-tranche strategy therefore sent a round number.

The size of the artefact, measured by retraining Phase 1 without the roundness features, is about **3
points of recall and 2 of PR-AUC** (0.548 → 0.516; 0.452 → 0.432), recorded in
[phase1.md](phase1.md). The same run also showed a second problem: a constant goal is a number the
retrained detector can memorise.

*Fix:* each episode now draws its own goal, log-uniformly in $50k–$400k, to the cent
(`EpisodeSampler.sample`). *Not fixed:* the generator still never produces round laundering. That is
the first thing to fix before real data, and it is why this phase closes the adversary's *access* to
the artefact rather than the artefact itself. The Phase 2 write-up also had to be corrected: it
claimed amount-weighting "guards against" the single giant transfer, which is only true if the
detector flags it.

### Run 2 — a decoder fingerprint (stopped partway)

With varied goals the picture flipped: after one round of feedback the detector caught nearly
everything (recall 0.96). That is too easy to believe. The decoder charged a fixed 0.2% fee and paid
everything in USD, so **every adversary row had `fx_loss` of exactly 0.002**, while benign traffic has
0 (93% of rows) or a 0.5–3% spread. A retrained detector can learn that constant instead of the
strategy.

It did. `fx_loss` was feature #56 of 74 in `D_0` (0.1% of gain) and #25 in run 2's `D_1` (1.0%). Re-scoring
the *same* transactions with benign-style currency fields dropped that `D_1`'s recall on `A_1` from **0.963
to 0.724**: about a quarter of the apparent catch was the fingerprint. The run was stopped rather than
finished.

*Fix, by the project's usual rule (make both paths share code):* the decoder's currency and
`amount_received` now come from the same `fx_legs` process that benign and background-laundering rows
use, and the extra fixed fee defaults to zero. `tests/test_adversary.py` pins the result. Phase 2 was
re-run on the corrected decoder; its headline moved from 0.767 → 0.247 to 0.750 → 0.231 (see
[phase2.md](phase2.md)), since `D_0` barely used `fx_loss`.

### Run 3 — noise, and a winner's curse

With both fixed the contest looked real, but one number stayed suspicious: `A_1`'s GA reported **best
fitness 0.999** while its held-out evasion was far lower. `A_1` turned out to be the single-transfer
strategy, which is a **coin flip per episode** — the detector flags that one transaction or it does
not, nothing in between (its per-episode evasion is bimodal: 0 or 1). Scoring thousands of coin-flip
genomes on 5 episodes each guarantees some score 5-for-5 by luck, and elitism keeps that lucky fitness
forever. The GA's "best fitness" is therefore optimistic for exactly the strategies that matter most.

The held-out measurement is the number to believe, and 30 coin-flip episodes cannot resolve it: the
standard error is about 0.09, and two honest samples of the same champion gave evasion readings 0.25
apart. `reevaluate_arena.py` re-scores every saved (detector, champion) pair on 200 episodes with a
bootstrap interval, **without re-running any GA**. That changed the middle rounds materially: the
30-episode readings (0.47, 0.47, 0.53) had flattered the detector against the 200-episode ones (0.28,
0.30, 0.37). The final table above uses the 200-episode numbers.

### A suspicion that did *not* pan out

`A_0` showed four different senders paying the same destination at the identical nanosecond — the
mechanism of Phase 2's tie exploit, on the receiver side. Re-scoring with every receipt spaced a
minute apart left detector recall **unchanged at 0.129**, so simultaneity is not what drives its
evasion, and no decoder change was made. Recorded because it was checked, and came back clean.

One more, found by a test rather than a run: an early version of the feedback pool counted its
per-round slice in *operations*, which are not equal in size (2 to 100+ rows), so one round's large
operations filled the cap and a later round got zero rows. It now counts rows
(`test_a_round_of_huge_operations_cannot_starve_the_others`).

## What this does and does not show

**It shows** that the closed loop works and its guards hold: the holdout is verified unchanged
(fingerprinted before and after), the alert budget is identical every round (invariant 3), the
feedback never exceeds its cap, and feedback measurably improves the detector against the strategies
it is shown, with only a small cost on clean data. It also shows that the evaluation machinery can
find real flaws, three of which are above.

**It does not show** that the detector can win, or that the race converges: the adversary still
succeeds in 56% of episodes at round 4, and the favourable trend is five rounds from one seed. It does
not show that results transfer to real data. And it should not be read as "the adversary got stronger":
the later strategies are not stronger against fixed targets (the upper triangle above).

## Limitations

1. **One run, one seed.** The intervals are over *episodes*, not over runs; run-to-run variance of the
   whole loop (a different GA seed) has not been measured. Differences of a few points between rounds
   are not evidence of anything, and the per-round "lesson" gains after the first are small enough
   that I would not defend their ordering.
2. **The feedback transfers imperfectly across time.** For `A_1` against `D_1`, the champion was
   flagged on 58% of episodes drawn from the period the detector trained and validated in, and only
   22% on later ones (60 episodes each; one champion; indicative, not established). Evaluating on later-period
   episodes, as done here, is the conservative choice; evaluating on episodes like those it trained on
   would have flattered the detector.
3. **The synthetic world is still the synthetic world.** Laundering is never round and never large
   (max $146k in training). The first is the artefact of Run 1; the second is why the single-large-
   transfer family persists. A detector trained on real data would not have either shortcut, and it
   would also not have the matching blind spot. Phase 1/2 numbers carry about +3 recall points from
   the roundness tell.
4. **The adversary reads the detector's exact scores**, which a real launderer cannot. That makes it
   stronger than a realistic attacker, so the detector's numbers are conservative on that axis.
5. **Feedback is positive-only and capped.** No benign look-alikes are synthesised, and the 20% cap
   bounds how far the detector can move; "the detector never wins" is partly a statement about the cap.
6. **Per-typology holdout recall is noisy.** `fan-in` recall slipped from 0.41 to 0.35 and
   `cycle` rose from 0.48 to 0.54, but each typology has on the order of a hundred holdout positives
   (standard error ≈ 0.05), so no typology claim is made.

## Artifacts

`reports/phase3_metrics.json` (per-round holdout metrics, champions, strategy pools, the cross-play
matrices with CI bounds), `phase3_arms_race.png`, `phase3_crossplay.png`, `phase3_typology_recall.png`,
`phase3_gene_drift.png`, `phase3_run.log`. The log is the original run's own output, which
measured on 30 episodes; the metrics and figures were then rewritten by `reevaluate_arena.py` with the
200-episode numbers quoted here, so the log's summary table will not match the report's. Detectors `D_0 … D_5` are saved to `artifacts/arena/`
(gitignored, regenerable). Code: `src/arena/loop/` (`pool`, `evaluation`, `orchestrator`,
`reevaluate`, `gate`, `report`); tests: `tests/test_loop.py`.

## What Phase 5 needs from this

- **Fix the generator's roundness leak first**, then re-run Phases 1–3: it is the one known artefact
  that still inflates the numbers.
- Add large-amount laundering to the generator, so "a lone large transfer" is something a detector can
  learn from and not a standing blind spot.
- Measure run-to-run variance (several GA seeds) before putting weight on any round-to-round effect.
- On real data, expect the first two artefacts to disappear and new ones to appear. Read the decoded
  transactions of any perfect score.
