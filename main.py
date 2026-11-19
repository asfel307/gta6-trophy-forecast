"""GTA VI trophy rarity forecaster — make.fun entry (Python 3.13, stdlib only).

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, its earn
rate on Exophase at the case's target, as five quantiles a trophy, scored by a
pinball loss on the logit scale.

What the data is
---------------
The case file gives, per trophy, every valid read the house made of
Exophase's trophy page before the checkpoint: the players tracked and that
trophy's earn rate (the percent of tracked players who hold it), oldest first.
Exophase refreshes a game's counts on its own schedule, hours apart, not
continuously: between refreshes the page is frozen, and at a refresh the
tracked base jumps and most rates move. Launch night shows the pattern
plainly — 638 players tracked, every rate unchanged for hours, then one
refresh to 3,868 players and every rate but the one at 100% lower, because
the newly tracked players hold fewer trophies.

Method, per trophy
------------------
Two views of where the rate is heading, blended 50/50 in logit space:

1. Trend. A recency-weighted least-squares line of logit(rate) on
   ln(1 + hours since release) over the last 24 hours of reads (all reads
   when the case spans less), read at the target. This is the house curve
   baseline's shape, but weighted toward recent reads and shrunk toward the
   last observed rate, because a line fitted over one day and read 4.5 days
   out overshoots when the launch-week curve bends.
2. Dilution. A least-squares line of the rate itself on 1 / players-tracked:
   the tracked base is a mixture of an early cohort and later joiners, so as
   the base grows the rate converges to the later joiners' rate. The
   players-tracked at the target comes from a log-log fit of players-tracked
   on ln(1 + hours), capped at 10x the last count, so the dilution view
   cannot run away when growth saturates.

The blend is shrunk 25% toward the last observed rate and its total move
from that rate is capped at 2.5 logit units. The spread is the trend fit's
residual spread combined with an extrapolation term (0.40 logit units per
unit of log-time beyond the last read) and a 0.30 structural floor for the
refresh jumps Exophase makes between reads, floored at 0.35 and capped at
1.60 logit units.

Stale page. When a trophy's rate never moves in the whole history (one
distinct value, as on launch night, when the page had not refreshed in
hours), there is no trend to fit: the page is stale and the next refresh is
expected to move the rate. The median is shifted down 0.30 logit units —
the refresh took every rate but the 100% one down — except a trophy already
at 100%, which is left where it is, and the spread is set to 0.85 logit
units. Flatness is judged over the whole history, not the fitting window:
a rate that converged to its level is flat in the window but moved a lot
getting there, and is not a stale page.

Assumptions, stated plainly
---------------------------
- GTA VI's trophy list is the case's own: 55 trophies, and the answer is the
  rates the page shows at the target, as of Exophase's last refresh before
  it. No trophy is added or renamed between the checkpoint and the target.
- The launch-week shape is a smooth curve in log-time with a dilution pull
  toward the marginal player's rate; neither view is trusted alone, hence
  the blend and the shrinkage.
- The refresh that straddles the practice horizon is a one-off level shift,
  not a trend: the stale-page branch handles it, and it never fires on a
  case whose history actually moves, which the final case's will.
- Rates are clamped to 0.01..99.99% before any logit, matching the scorer.

Deterministic: no randomness, no clock, no network; the same case file
always writes the same bytes. One case runs in well under a second.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only: no network, no pip packages. -P
keeps this file's directory OFF sys.path, so a module shipped beside main.py
imports only after the sys.path line below, which puts the directory first;
-s drops user site-packages.

What an entry must do:
  - read every <id>.in in the input directory (the grader runs one case at a
    time, with only that case's file there): JSON {"schema":
    "makefun/gta6-trophies-case/v1", "id", "at", "horizon_minutes", "target",
    "levels", "game", "trophies", "history", "notes"}, where `trophies`
    lists the case's trophies {id, name, description, tier, points, secret}
    and `history` every read before `at`, oldest first, each {slot, at,
    players_tracked, trophies_earned, rates}, `rates` in the order of
    `trophies` (null where that read lacked the trophy);
  - write <id>.out to the output directory: {"trophies": [{"id": 1, "q":
    [q05, q25, q50, q75, q95]}, ...]}, one row for each of the case's
    trophies, each id once, no other key, each q five finite numbers from 0
    to 100 (percent), non-decreasing; at most 65,536 bytes. Anything else is
    an invalid case, scored as if every quantile equalled the trophy's rate
    in the last read, and an entry with an invalid case takes no prize.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# GTA VI's release, the origin of the log-time axis (the first reads are
# dated the night before, 2026-11-18T23:50Z, so hours since release start
# just below zero and are floored at zero).
RELEASE = datetime(2026, 11, 19, tzinfo=timezone.utc)

# --- tuning constants (see the module docstring for what each one means) ---
# Reads used for the fits: the last 24 hours before the checkpoint.
WINDOW_HOURS = 24.0
# Recency half-life for the trend fit's weights, in hours.
HALF_LIFE_HOURS = 12.0
# Blend weight of the trend view against the dilution view.
W_TREND = 0.5
# Shrinkage of the blended median toward the last observed rate.
SHRINK = 0.25
# Largest allowed move of the median from the last rate, in logit units.
MAX_MOVE = 2.5
# Largest allowed growth of the tracked base beyond the last read.
MAX_N_GROWTH = 10.0
# Stale-page branch: median shift and spread, in logit units.
STALE_SHIFT = -0.30
STALE_SIGMA = 0.85
# Spread floor, cap, structural floor and extrapolation rate, logit units.
SIGMA_FLOOR = 0.35
SIGMA_CAP = 1.60
SIGMA_BASE = 0.30
SIGMA_EXTRAP = 0.40
# A window counts as flat when it holds at most this many distinct rates
# or its logit range is under this many units.
FLAT_DISTINCT = 2
FLAT_RANGE = 0.05


def clamp_rate(pct: float) -> float:
    """A percent clamped to the range the logit is defined on."""
    return min(99.99, max(0.01, pct))


def logit(pct: float) -> float:
    """logit of a percent, after clamping."""
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(y: float) -> float:
    """A logit back to a percent, clamped to the scored range."""
    if y >= 0:
        p = 1.0 / (1.0 + math.exp(-y))
    else:
        e = math.exp(y)
        p = e / (1.0 + e)
    return clamp_rate(100.0 * p)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's Math.round(x * 10000) / 10000 rounds it.

    A whole number comes back as an int, so json.dumps writes 100 where JavaScript writes 100 (not 100.0).
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(text: str) -> datetime:
    """An ISO 8601 UTC timestamp from the case files, as a datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_release(moment: datetime) -> float:
    """Hours from release to a read, floored at zero (launch-night reads)."""
    return max(0.0, (moment - RELEASE).total_seconds() / 3600.0)


def median(values: list[float]) -> float:
    """The median of a non-empty list of numbers (deterministic)."""
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def wols(xs: list[float], ys: list[float], ws: list[float]) -> tuple[float, float, float]:
    """Weighted least-squares line y = a + b*x; returns (a, b, residual std).

    The residual std is the weighted root-mean-square of the residuals, so
    it is zero when the points sit exactly on the line (a flat history) and
    grows with the scatter the line cannot explain.
    """
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx == 0:
        b = 0.0
    else:
        b = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys)) / sxx
    a = my - b * mx
    resid2 = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys)) / sw
    return a, b, math.sqrt(max(0.0, resid2))


def ols(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Unweighted least-squares line y = a + b*x; returns (a, b)."""
    return wols(xs, ys, [1.0] * len(xs))[:2]


def trophy_series(case: dict, index: int) -> list[tuple[float, float | None, float]]:
    """One trophy's usable history: (hours since release, players, rate).

    Reads with no rate for this trophy drop out; the list stays in time
    order, oldest first, as the case file holds it.
    """
    series = []
    for read in case.get("history") or []:
        rates = read.get("rates") or []
        if index >= len(rates):
            continue
        rate = rates[index]
        if rate is None:
            continue
        try:
            moment = parse_time(read["at"])
        except (KeyError, TypeError, ValueError):
            continue
        players = read.get("players_tracked")
        if players is not None:
            try:
                players = float(players)
            except (TypeError, ValueError):
                players = None
        series.append((hours_since_release(moment), players, float(rate)))
    return series


def forecast_trophy(series: list[tuple[float, float | None, float]], x_target: float) -> tuple[float, float]:
    """One trophy's (median logit, spread in logit units) at the target.

    `x_target` is ln(1 + hours from release to the target). Returns the
    median as a logit and the spread as a standard deviation in logit units;
    the caller turns them into the five quantiles.
    """
    last_rate = series[-1][2]
    y_last = logit(last_rate)
    t_last = series[-1][0]
    x_last = math.log1p(t_last)

    # The fitting window: the last 24 hours of reads, or all of them when
    # the case spans less (the practice cases span about three hours).
    window = [row for row in series if row[0] >= t_last - WINDOW_HOURS]
    if len(window) < 3:
        window = series
    # Flatness is judged over the WHOLE history, not the window: a rate
    # that converged to its level (a story trophy the whole base has
    # finished, a rare one diluted to nothing) is flat in the window but
    # moved a lot getting there, and is not a stale page. Only a page that
    # never moved at all — every read identical, as on launch night — is
    # stale, and only then is the next refresh expected to move the rate.
    all_logits = [logit(row[2]) for row in series]
    flat = len({row[2] for row in series}) <= FLAT_DISTINCT or (
        max(all_logits) - min(all_logits)
    ) < FLAT_RANGE

    if flat or len(window) < 3:
        # Stale page: no trend to fit, so the next refresh is expected to
        # move the rate down (the newly tracked players hold fewer
        # trophies), except a trophy already at 100%.
        if last_rate >= 99.5:
            return y_last, STALE_SIGMA
        return y_last + STALE_SHIFT, STALE_SIGMA

    xs = [math.log1p(row[0]) for row in window]
    ys = [logit(row[2]) for row in window]
    ws = [0.5 ** ((t_last - row[0]) / HALF_LIFE_HOURS) for row in window]

    # View 1: the log-time trend, read at the target.
    a_trend, b_trend, resid = wols(xs, ys, ws)
    y_trend = a_trend + b_trend * x_target

    # View 2: dilution toward the marginal player's rate. Fit the rate on
    # 1 / players-tracked, forecast the tracked base at the target from its
    # own log-log growth, and read the dilution line there.
    y_dil = None
    ns = [(row[0], row[1], row[2]) for row in window if row[1] is not None and row[1] > 0]
    if len(ns) >= 3 and len({n for _, n, _ in ns}) >= 2:
        a_dil, b_dil = ols([1.0 / n for _, n, _ in ns], [r for _, _, r in ns])
        n_last = series[-1][1]
        if n_last is not None and n_last > 0:
            xs_n = [math.log1p(t) for t, _, _ in ns]
            ws_n = [0.5 ** ((t_last - t) / HALF_LIFE_HOURS) for t, _, _ in ns]
            ln_n = [math.log(n) for _, n, _ in ns]
            a_n, b_n, _ = wols(xs_n, ln_n, ws_n)
            ln_n_target = a_n + b_n * x_target
            ln_n_target = min(ln_n_target, math.log(n_last) + math.log(MAX_N_GROWTH))
            n_target = math.exp(ln_n_target)
            r_dil = a_dil + b_dil / n_target
            y_dil = logit(r_dil)

    if y_dil is None:
        y_med = y_trend
    else:
        y_med = W_TREND * y_trend + (1.0 - W_TREND) * y_dil

    # Shrink toward the last observed rate and cap the total move: a line
    # fitted over one day and read 4.5 days out must not run away.
    y_med = (1.0 - SHRINK) * y_med + SHRINK * y_last
    move = max(-MAX_MOVE, min(MAX_MOVE, y_med - y_last))
    y_med = y_last + move

    # Spread: the fit's residual spread, an extrapolation term for the
    # log-time distance beyond the last read, and a structural floor for
    # the refresh jumps between reads.
    dx = max(0.0, x_target - x_last)
    sigma = math.sqrt(resid * resid + (SIGMA_EXTRAP * dx) ** 2 + SIGMA_BASE ** 2)
    sigma = min(SIGMA_CAP, max(SIGMA_FLOOR, sigma))
    return y_med, sigma


def forecast_case(case: dict, name: str) -> dict:
    """The forecast for one case: {"trophies": [{"id": n, "q": [...]}, ...]}."""
    levels = case.get("levels") or LEVELS
    if list(levels) != list(LEVELS):
        raise ValueError(f"{name} asks for levels {levels}, not {LEVELS}")
    history = case.get("history") or []
    try:
        target = parse_time(case["target"])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"{name} has no usable target")
    x_target = math.log1p(hours_since_release(target))

    rows = []
    for index, trophy in enumerate(case["trophies"]):
        series = trophy_series(case, index)
        if not series:
            # No read ever listed this trophy: a flat 1% with the widest
            # spread, so the case still validates instead of crashing.
            y_med, sigma = logit(1.0), SIGMA_CAP
        else:
            y_med, sigma = forecast_trophy(series, x_target)
        quantiles = [round4(inv_logit(y_med + z * sigma)) for z in Z]
        for i in range(1, len(quantiles)):
            quantiles[i] = max(quantiles[i], quantiles[i - 1])
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
