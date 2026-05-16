# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Methodology: Matteo Conti (matfinog) quantitative approach

All quant strategies in this repo follow this framework:

**Data**: Dukascopy tick data (bid/ask) via Tickstory. Always export in pure UTC. Raw CSV format: `Date,Timestamp,Bid Price,Ask Price,Last Price,Volume`.

**Backtesting stack**: NautilusTrader (Python/Rust) for final validation. Vectorized pandas for fast optimization grid search. Never use NautilusTrader inside optimization loops — too slow.

**Standard pipeline per strategy:**
1. `src/pipeline.py` — raw CSV → M5 parquet (OHLCV + spread + bar_delta) + NautilusTrader QuoteTick catalog
2. `backtests/optimize.py` — MT5-style grid optimizer: user edits `OPTIMIZE = {"param": (start, step, stop)}` ranges
3. `backtests/run_backtest.py` — NautilusTrader single run with realistic FillModel (slippage + spread)
4. `analysis/montecarlo.py` — 1000 shuffle-trade permutations to test edge robustness

## Anti-overfitting rules (non-negotiable)

- **IS/OOS hard split**: In-sample optimization on first ~80% of data. Out-of-sample (last ~20%) touched exactly once, after final parameter selection.
- **Robust zone selection**: Never pick single best parameter combination. Identify clusters where nearby params also perform well.
- **Monte Carlo validation**: Before touching OOS, run shuffle Monte Carlo. If P(EV≤0) > 0.05 → edge not statistically significant.
- **Minimum trades**: Combinations with < 10 trades in IS period → discard (too few for statistics).

## Tech stack for quant strategies

```
nautilus_trader   # backtesting engine (event-driven, realistic fills)
pandas / numpy    # vectorized data processing
pandas_ta         # technical indicators
pyarrow           # parquet I/O
matplotlib        # plots
scipy             # statistics
```

Install: `pip install -r requirements.txt`

## Existing data (Trading/XAU/)

XAUUSD OHLCV CSVs available locally: M1, M5, M15, M30, H1, H4, D1. No bid/ask — spread must be approximated if used. Useful for quick indicator testing, not for realistic spread/slippage backtests.

## Project structure convention

Each strategy lives in its own folder with a `CLAUDE.md` describing strategy-specific logic. This parent `CLAUDE.md` covers only shared methodology.
