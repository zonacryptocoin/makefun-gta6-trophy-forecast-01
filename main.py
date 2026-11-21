"""GTA VI trophy rarity forecast -- make.fun gta6-trophy-rarity-forecast-01.

Forecasts every PS5 trophy's Exophase earn rate at the case target, as five
quantiles per trophy, from the case's own read history. Python 3.13,
standard library only, deterministic, no network.

Method (per trophy, in logit space):
  - Collapse the history to distinct snapshots. Exophase refreshes its
    counts in steps, so most consecutive reads are identical; a plateau
    contributes one point, timed at its last read.
  - Fit logit(rate) against ln(1 + hours since launch) by recency-weighted
    least squares (half-life 36 h), so the line tracks the recent trend
    rather than the pre-launch cohort.
  - Shrink each trophy's slope toward the pooled slope of its class
    (story / activity / grind, from the trophy's own description). A single
    trophy's short, noisy series overfits on its own; grind trophies
    (100% completion, 250 stunts, card sets, ...) barely move in week one,
    while story trophies climb fast.
  - Spread = the fit residual (floored by the pooled residual) combined
    with a random-walk term that grows with the horizon, plus a small
    allowance for an Exophase refresh landing between the last read and the
    target. A refresh dilutes rates (new players join faster than they
    earn), so the allowance fattens the downside tail, most when the
    history shows no refresh yet.
  - Medians and quantiles are inverse-logit transforms, clamped to
    0.01..99.99% before the logit, rounded to 4 decimals half-up.

How the grader runs it: `python3 -P -s /bot/main.py` in python:3.13-slim.
It reads every <id>.in in the input directory and writes <id>.out to the
output directory. Outside the grader, MAKEFUN_INPUT_DIR and
MAKEFUN_OUTPUT_DIR stand in for /input and /output.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Descriptions of trophies that barely move in week one: completionist
# counts and sets. Everything secret is a story/mission trophy; the rest
# that is not grind is a one-off activity.
GRIND_MARKS = (
    "100% completion",
    "250 stunts",
    "card sets",
    "100 locations",
    "50 challenges",
    "100 vehicles",
    "20 different races",
    "every location",
    "every spot",
    "all garage",
    "all shooting range",
)

HALF_LIFE_HOURS = 36.0
RW_SIGMA = 0.5          # logit units per sqrt(day), the house baseline's prior
RESID_FLOOR = 0.25      # logit units; a flat series still has refresh noise
SPREAD_FLOOR = 0.30
SPREAD_CAP = 2.5
SLOPE_CAP = 2.5
REFRESH_LOGIT = 1.5     # typical logit drop when Exophase refreshes a flat page
                        # (raised from 0.9 on 2026-11-21: validation showed the
                        # downside tail was too thin for diluted answers --
                        # practice f<1 mean loss 0.5394 -> 0.4870, f=1.0
                        # 0.0685 -> 0.0827, synthetic final also slightly
                        # better; medians untouched, only s_down widens)
REFRESH_EVERY_HOURS = 5.0


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg: float) -> float:
    lg = max(-30.0, min(30.0, lg))
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's Math.round does it.

    A whole number comes back as an int, so json.dumps writes 100 where
    JavaScript writes 100 (not 100.0).
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def classify(trophy: dict) -> str:
    desc = trophy["description"].lower()
    if any(mark in desc for mark in GRIND_MARKS):
        return "grind"
    if trophy.get("secret"):
        return "story"
    return "activity"


def weighted_ols(xs: list[float], ys: list[float], ws: list[float]) -> tuple[float, float]:
    """Recency-weighted least squares for y ~ a + b*x. Returns (a, b)."""
    sw = sum(ws)
    if len(xs) < 2 or sw <= 0:
        mean_y = sum(w * y for w, y in zip(ws, ys)) / sw if sw > 0 else 0.0
        return mean_y, 0.0
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return my, 0.0
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    return my - b * mx, b


def collapse(series: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Collapse consecutive duplicate rates to one point, timed at the
    plateau's last read. `series` is (hours_since_launch, rate) oldest first.
    """
    runs: list[tuple[float, float]] = []
    for t, rate in series:
        if runs and runs[-1][1] == rate:
            runs[-1] = (t, rate)
        else:
            runs.append((t, rate))
    return runs


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    trophies = case["trophies"]
    history = case.get("history") or []
    target_dt = parse_time(case["target"])
    t_target = (target_dt - LAUNCH).total_seconds() / 3600.0
    x_target = math.log(max(1.0 + t_target, 0.05))
    horizon_days = case["horizon_minutes"] / 1440.0
    horizon_hours = case["horizon_minutes"] / 60.0

    # Per-trophy distinct snapshots: (x, y, w) with x = ln(1 + hours since
    # launch), y = logit(rate), w = recency weight.
    per_trophy: list[dict] = []
    for i, trophy in enumerate(trophies):
        raw: list[tuple[float, float]] = []
        for read in history:
            rates = read.get("rates") or []
            if i >= len(rates) or rates[i] is None:
                continue
            slot = parse_time(read["slot"])
            t = (slot - LAUNCH).total_seconds() / 3600.0
            raw.append((t, float(rates[i])))
        runs = collapse(raw)
        t_last = runs[-1][0] if runs else 0.0
        xs = [math.log(max(1.0 + t, 0.05)) for t, _ in runs]
        ys = [logit(rate) for _, rate in runs]
        ws = [0.5 ** ((t_last - t) / HALF_LIFE_HOURS) for t, _ in runs]
        per_trophy.append({"trophy": trophy, "runs": runs, "xs": xs, "ys": ys, "ws": ws})

    # First pass: each trophy's own slope, then the pooled slope per class
    # and the pooled residual across trophies.
    own_slopes: list[float] = []
    own_resids: list[float] = []
    for entry in per_trophy:
        xs, ys, ws = entry["xs"], entry["ys"], entry["ws"]
        a, b = weighted_ols(xs, ys, ws)
        entry["own_a"], entry["own_b"] = a, b
        if len(xs) >= 3 and (max(ys) - min(ys)) > 0.05:
            own_slopes.append(min(max(b, 0.0), SLOPE_CAP))
        if len(xs) >= 2:
            resid = math.sqrt(
                sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys)) / sum(ws)
            )
            own_resids.append(resid)
    pooled_resid = median(own_resids) if own_resids else RESID_FLOOR
    pooled_resid = min(max(pooled_resid, RESID_FLOOR), 1.5)

    pool: dict[str, float] = {}
    for cls in ("story", "activity", "grind"):
        slopes = [
            min(max(entry["own_b"], 0.0), SLOPE_CAP)
            for entry in per_trophy
            if classify(entry["trophy"]) == cls
            and len(entry["xs"]) >= 3
            and (max(entry["ys"]) - min(entry["ys"])) > 0.05
        ]
        pool[cls] = median(slopes) if slopes else 0.0

    rw_term = RW_SIGMA * math.sqrt(max(horizon_days, 0.0))
    p_refresh = min(1.0, horizon_hours / REFRESH_EVERY_HOURS)

    rows = []
    for entry in per_trophy:
        trophy = entry["trophy"]
        xs, ys, ws = entry["xs"], entry["ys"], entry["ws"]
        cls = classify(trophy)
        if not xs:
            # No read ever listed this trophy: low median, wide band.
            med, s_up, s_down = 1.0, 2.0, 2.0
        else:
            reliable = len(xs) >= 3 and (max(ys) - min(ys)) > 0.05
            if reliable:
                b = 0.7 * entry["own_b"] + 0.3 * pool[cls]
            elif cls == "grind":
                b = 0.3 * pool[cls]
            else:
                b = pool[cls]
            b = min(max(b, 0.0), SLOPE_CAP)
            # Refit the intercept with the shrunk slope, so the line tracks
            # the recent level.
            a = sum(w * (y - b * x) for w, x, y in zip(ws, xs, ys)) / sum(ws)
            if len(xs) >= 2:
                s_resid = math.sqrt(
                    sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys)) / sum(ws)
                )
            else:
                s_resid = pooled_resid
            s_resid = min(max(s_resid, pooled_resid, RESID_FLOOR), 1.5)
            s_up = math.sqrt(s_resid ** 2 + rw_term ** 2)
            flat = len(xs) <= 2
            allowance = REFRESH_LOGIT * p_refresh * (1.0 if flat else 0.25)
            s_down = math.sqrt(s_up ** 2 + allowance ** 2)
            s_up = min(max(s_up, SPREAD_FLOOR), SPREAD_CAP)
            s_down = min(max(s_down, SPREAD_FLOOR), SPREAD_CAP)
            med = clamp_pct(inv_logit(a + b * x_target))
        base = logit(med)
        spreads = [s_down, s_down, s_up, s_up, s_up]
        qs: list[float | int] = []
        for z, s in zip(Z, spreads):
            q = round4(clamp_pct(inv_logit(base + z * s)))
            if qs:
                q = max(q, qs[-1])
            qs.append(q)
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
