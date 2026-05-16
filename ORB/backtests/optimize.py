"""
optimize.py — Grid optimizer stile MT5

Modifica la sezione PARAMETRI come faresti nel tab "Inputs" di MT5:
  "nome_param": (start, step, stop)   ← stop INCLUSO

Risultati salvati in backtests/results/optimization.csv
ordinati per net_profit.

Nota: usa backtester vettoriale (pandas) per velocità.
      NautilusTrader usato solo per validazione finale dei migliori params.
"""

import itertools
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETRI DA OTTIMIZZARE  — (start, step, stop)  stop è INCLUSO
# Metti None al posto della tupla per FISSARE un parametro al valore in FIXED
# ═══════════════════════════════════════════════════════════════════════════════

OPTIMIZE = {
    # (start, step, stop) — stop INCLUSO
    "orb_start_utc_hour": (6,   1,  14),    # 6 → 14 UTC, step 1h
    "orb_duration_min":   (15, 15,  90),    # 15 → 90 min, step 15
    "tp_rr_ratio":        (0.5, 0.25, 3.0), # TP = entry + ratio*(entry - ORB_Low)
    "delta_threshold":    (0,   50, 400),   # 0 = nessun filtro delta
}

# Parametri fissi
FIXED = {
    "orb_start_utc_min":   0,
    "trade_end_utc_hour": 19,   # 19 UTC = 14:00 ET winter
}

# Periodo IN-SAMPLE — non toccare OOS (2023+)
IS_START = "2015-01-01"
IS_END   = "2022-12-31"

# Path dati M5 (costruito da pipeline.py)
M5_PATH  = Path(__file__).parent.parent / "data" / "m5_xauusd.parquet"
OUT_CSV  = Path(__file__).parent / "results" / "optimization.csv"

# ═══════════════════════════════════════════════════════════════════════════════


def make_grid(optimize: dict) -> list[dict]:
    """Genera tutte le combinazioni dai range definiti."""
    keys   = list(optimize.keys())
    ranges = []
    for k, (start, step, stop) in optimize.items():
        n = round((stop - start) / step) + 1
        vals = [round(start + i * step, 8) for i in range(n)]
        ranges.append(vals)

    combos = list(itertools.product(*ranges))
    grid   = [dict(zip(keys, c)) for c in combos]
    return grid


def vectorized_backtest(m5: pd.DataFrame, params: dict) -> dict | None:
    """Backtest ORB vettoriale su M5 OHLCV. Ritorna metriche o None."""

    orb_h    = int(params["orb_start_utc_hour"])
    orb_m    = int(FIXED["orb_start_utc_min"])
    dur      = int(params["orb_duration_min"])
    rr       = params["tp_rr_ratio"]
    delta_th = params["delta_threshold"]
    close_h  = int(FIXED["trade_end_utc_hour"])

    has_delta = "bar_delta" in m5.columns

    trades = []

    for date, day in m5.groupby(m5.index.date):
        orb_start_min = orb_h * 60 + orb_m
        orb_end_min   = orb_start_min + dur

        day_min = day.index.hour * 60 + day.index.minute

        orb_bars = day[(day_min >= orb_start_min) & (day_min < orb_end_min)]
        if len(orb_bars) < 1:
            continue

        orb_high = orb_bars["high"].max()
        orb_low  = orb_bars["low"].min()

        trigger_bars = day[(day_min >= orb_end_min) & (day_min < close_h * 60)]
        if len(trigger_bars) < 1:
            continue

        pnl      = None
        entry    = None
        tp_price = None
        sl_price = None
        in_trade = False

        for _, bar in trigger_bars.iterrows():
            if not in_trade:
                # Long only: close must break above ORB_High
                if bar["close"] <= orb_high:
                    continue

                # Delta filter (approssimato da tick direction)
                if has_delta and delta_th > 0:
                    if bar.get("bar_delta", 0) < delta_th:
                        continue

                entry    = bar["close"] + (bar["spread"] / 2)  # fill at ask
                rng      = entry - orb_low
                if rng <= 0:
                    continue
                tp_price = entry + rr * rng
                sl_price = orb_low
                in_trade = True

            else:
                # Check TP/SL on this bar
                if bar["high"] >= tp_price:
                    pnl = tp_price - entry
                    break
                if bar["low"] <= sl_price:
                    pnl = sl_price - entry
                    break

        # EOD close if trade open
        if in_trade and pnl is None:
            exit_px = trigger_bars.iloc[-1]["close"]
            pnl = exit_px - entry

        if pnl is not None:
            trades.append(pnl)

    if len(trades) < 10:
        return None

    pnls   = np.array(trades)
    net    = pnls.sum()
    n      = len(pnls)
    wr     = (pnls > 0).mean()
    avg    = pnls.mean()
    cumsum = np.cumsum(pnls)
    peak   = np.maximum.accumulate(cumsum)
    max_dd = (peak - cumsum).max()

    return {
        "net_profit":  round(net, 2),
        "n_trades":    n,
        "win_rate":    round(wr * 100, 1),
        "avg_trade":   round(avg, 2),
        "max_dd":      round(max_dd, 2),
    }


def run():
    # ── Carica M5 ─────────────────────────────────────────────────────────────
    if not M5_PATH.exists():
        print(f"M5 file non trovato: {M5_PATH}")
        print("Esegui prima: python src/pipeline.py")
        return

    print(f"Carico M5 in-sample ({IS_START} → {IS_END})...")
    m5 = pd.read_parquet(M5_PATH)
    m5 = m5[(m5.index >= IS_START) & (m5.index <= IS_END)]
    print(f"  {len(m5):,} barre M5 caricate")

    # ── Genera griglia ────────────────────────────────────────────────────────
    grid = make_grid(OPTIMIZE)
    print(f"\nCombinazioni totali: {len(grid):,}")
    print("Avvio ottimizzazione...\n")

    # ── Esegui ────────────────────────────────────────────────────────────────
    results = []
    t0 = datetime.now()

    for i, params in enumerate(grid):
        res = vectorized_backtest(m5, params)
        if res:
            row = {**params, **res}
            results.append(row)

        # Progress ogni 5%
        if (i + 1) % max(1, len(grid) // 20) == 0:
            pct     = (i + 1) / len(grid) * 100
            elapsed = (datetime.now() - t0).seconds
            eta_s   = elapsed / (i + 1) * (len(grid) - i - 1)
            print(f"  {pct:.0f}%  [{i+1}/{len(grid)}]  ETA {eta_s/60:.1f} min", end="\r")

    print(f"\n\nCompletato in {(datetime.now()-t0).seconds/60:.1f} min")
    print(f"Combinazioni con trade: {len(results):,} / {len(grid):,}")

    # ── Salva risultati ───────────────────────────────────────────────────────
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results).sort_values("net_profit", ascending=False)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nRisultati → {OUT_CSV}")

    # ── Top 20 ───────────────────────────────────────────────────────────────
    print("\n── TOP 20 per net_profit ──────────────────────────────────────────")
    cols = ["orb_start_utc_hour", "orb_duration_min",
            "tp_rr_ratio", "delta_threshold",
            "net_profit", "n_trades", "win_rate", "avg_trade", "max_dd"]
    print(df[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    run()
