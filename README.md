# makefun-gta6-trophy-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every PlayStation 5 trophy's earn rate on Exophase for Grand Theft
Auto VI at 2026-11-26T00:00:00Z, as five quantiles per trophy, from the
trophy page's own read history in the case file. The prize is $250 in USDC
(20 places paid, 1st $50); a forecast must beat the best house entry to
take a paid place.

## Method

`main.py` works per trophy, in logit space, because earn rates live in
[0, 100] and a miss from 1% to 2% costs about what a miss from 33% to 50%
does:

- The history is collapsed to distinct snapshots — Exophase refreshes its
  counts in steps, so most consecutive reads are identical.
- `logit(rate)` is fit against `ln(1 + hours since launch)` by
  recency-weighted least squares (half-life 36 h), so the line tracks the
  recent trend, not the tiny pre-launch cohort.
- Each trophy's slope is shrunk toward the pooled slope of its class
  (story / activity / grind, read from the trophy's own description).
  Grind trophies (100% completion, 250 stunts, card sets, ...) barely move
  in week one; story trophies climb fast. A single trophy's short, noisy
  series overfits on its own.
- The spread is the fit residual (floored by the pooled residual) combined
  with a random-walk term that grows with the horizon, plus a small
  allowance for an Exophase refresh landing between the last read and the
  target. A refresh dilutes rates, so the allowance fattens the downside
  tail, most when the history shows no refresh yet.

Python 3 standard library only, deterministic, no network. It reads every
`<id>.in` in the input directory and writes `<id>.out` to the output
directory.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=5j9p514c&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
