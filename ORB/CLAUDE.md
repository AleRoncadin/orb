# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Strategy: Fabio Valentini ORB (validated by Matteo Conti)

Source of truth: `Fabio IVB Model, The Institutional Protocol.pdf`  
`matteo_conti_channel.txt` is the channel of an institution trader.

Core rules (never change without explicit user instruction):
- **LONG ONLY** — no shorts, no stop & reverse
- **SL = ORB_Low** (dynamic per day, not fixed USD)
- **TP = entry + tp_rr_ratio × (entry − ORB_Low)** (range-scaled, parametric)
- **No body/wick filter** — that was a NotebookLM invention
- **Delta filter**: BarDelta ≥ delta_threshold (approximated from tick direction)
- **1 trade per session**, EOD force-close at `trade_end_utc_hour`
- All timestamps UTC — no DST, no ET conversion in code

## Pipeline (run in order)

```bash
# 1. Build M5 parquet from Tickstory/Dukascopy CSV
python src/pipeline.py

# 2. Run single NautilusTrader backtest (in-sample 2015–2022)
python backtests/run_backtest.py

# 3. Grid optimization (vectorized pandas, fast)
python backtests/optimize.py

# 4. Monte Carlo robustness analysis
python analysis/montecarlo.py
```

## Data

- **Source**: Dukascopy tick data via Tickstory (select UTC, not UTC+2)
- **Raw CSV**: `data/raw/XAUUSD.csv` — format: `Date,Timestamp,Bid Price,Ask Price,Last Price,Volume`
- **M5 parquet**: `data/m5_xauusd.parquet` — columns: open, high, low, close, spread, bar_delta
- **NautilusTrader catalog**: `data/catalog/` — QuoteTick objects for run_backtest.py

bar_delta = sum of tick directions per M5 bar (Last ≥ Ask → +1, Last ≤ Bid → −1).

## IS/OOS Split — critical anti-overfitting rule

| Period | Use |
|---|---|
| 2015–2022 | In-sample: optimization + backtest |
| 2023–2025 | Out-of-sample: **never touch during optimization** |

OOS is validated only once, on the final selected parameters.

## Architecture

**`src/pipeline.py`** — CSV → artifacts:
- `run()`: writes QuoteTick catalog for NautilusTrader
- `build_m5()`: writes M5 parquet for optimizer (faster than NautilusTrader per combination)

**`src/strategy_orb.py`** — NautilusTrader strategy:
- `ORBConfig`: frozen Pydantic config (all optimizable params)
- `ORBStrategy.on_bar()`: ORB accumulation → entry check → EOD close
- `_bar_delta` not auto-populated from NautilusTrader bars — requires custom tick handler if delta filter is needed in live/backtest mode

**`backtests/optimize.py`** — MT5-style grid optimizer:
- Edit only the `OPTIMIZE` dict: `"param": (start, step, stop)` (stop inclusive)
- Edit `FIXED` for parameters held constant
- Uses vectorized pandas backtest (not NautilusTrader) for speed
- Results → `backtests/results/optimization.csv` sorted by net_profit
- NautilusTrader used only for final validation of best params

**`backtests/run_backtest.py`** — NautilusTrader single run:
- FillModel: prob_slippage=0.4, random_seed=42
- Starting balance: $100,000
- Outputs trades CSV to `backtests/results/trades_<label>.csv`

**`analysis/montecarlo.py`** — Matteo Conti shuffle method:
- 1000 permutations of trade sequence
- Tests if edge is sequence-dependent (fragile) or structural (robust)
- Input: `backtests/results/trades_<label>.csv` with `realized_pnl` column
- Output: PNG plots in `analysis/plots/`

## Parameter selection workflow (anti-overfitting)

1. Run `optimize.py` on IS data
2. Look for **robust zones** (clusters of good params), not single best combination
3. Validate top candidates with `run_backtest.py` (NautilusTrader, realistic fills)
4. Run `montecarlo.py` on best candidate trades
5. Only then: test on OOS (2023–2025), one time
