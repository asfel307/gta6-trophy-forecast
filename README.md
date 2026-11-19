# GTA VI trophy rarity forecast — make.fun entry

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/asfel307/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/asfel307/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, the
percent of the players Exophase tracks who will have earned it at
2026-11-26T00:00:00Z, a week after launch, as five quantiles a trophy,
scored by a pinball loss on the logit scale.

## Method (`main.py`, Python 3.13 stdlib only)

Exophase refreshes a game's counts on its own schedule, hours apart: between
refreshes the page is frozen, and at a refresh the tracked base jumps and
most rates move. Launch night shows it plainly — 638 players tracked, every
rate unchanged for hours, then one refresh to 3,868 players and every rate
but the one at 100% lower, because the newly tracked players hold fewer
trophies.

Per trophy, two views of where the rate is heading, blended 50/50 in logit
space:

1. **Trend.** A recency-weighted least-squares line of `logit(rate)` on
   `ln(1 + hours since release)` over the last 24 hours of reads, read at
   the target — the house curve baseline's shape, weighted toward recent
   reads and shrunk 25% toward the last observed rate, with the total move
   capped at 2.5 logit units.
2. **Dilution.** A least-squares line of the rate on `1 / players-tracked`:
   the tracked base is a mixture of an early cohort and later joiners, so as
   the base grows the rate converges to the later joiners' rate. The tracked
   base at the target comes from a log-log fit of its growth, capped at 10x
   the last count.

Spread: the trend fit's residual spread combined with an extrapolation term
(0.40 logit units per unit of log-time beyond the last read) and a 0.30
structural floor for the refresh jumps, floored at 0.35 and capped at 1.60.

**Stale page.** When a trophy's rate never moves in the window (one distinct
value, as on launch night), there is no trend to fit: the median is shifted
down 0.30 logit units — the refresh took every rate but the 100% one down —
except a trophy already at 100%, and the spread is 0.85 logit units.

Assumptions are stated in full in `main.py`'s docstring. Deterministic, no
network, ~0.1 s per case.

## Local practice check

The kit ships no local grader — the scorer and reference data stay on the
server — so there is no local practice loss to quote. What was checked
locally on the kit's practice cases p00/p01 (run twice, byte-identical
output, both cases valid):

- every trophy's median sits at or just below its last rate with the
  stale-page spread (the histories are flat launch night, so the stale-page
  branch fires, as designed);
- the 100% trophy stays at 100%;
- output is valid JSON, 55 rows, five non-decreasing quantiles each, under
  the 65,536-byte cap.

The house persistence baseline scores about 1.51 on the practice; this
entry's practice score is read from the server grade after submit.

## Run locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=8htny0qp&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
