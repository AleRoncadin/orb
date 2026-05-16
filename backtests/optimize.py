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
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# fix Windows console UTF-8
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

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


def prepare_daily_data(m5: pd.DataFrame) -> list:
    """Pre-split M5 in liste numpy per giorno. Chiamato una volta sola."""
    bar_min = (m5.index.hour * 60 + m5.index.minute).values
    has_delta = "bar_delta" in m5.columns

    highs   = m5["high"].values
    lows    = m5["low"].values
    closes  = m5["close"].values
    spreads = m5["spread"].values
    deltas  = m5["bar_delta"].values if has_delta else None

    days = []
    # group by date using normalized index (fast, no Python datetime objects)
    date_ints = (m5.index.year * 10000 + m5.index.month * 100 + m5.index.day).values
    for d in np.unique(date_ints):
        mask = date_ints == d
        days.append({
            "minutes": bar_min[mask],
            "high":    highs[mask],
            "low":     lows[mask],
            "close":   closes[mask],
            "spread":  spreads[mask],
            "delta":   deltas[mask] if deltas is not None else None,
        })
    return days


def vectorized_backtest(days: list, params: dict) -> dict | None:
    """Backtest ORB su dati pre-splittati (numpy puro). Ritorna metriche o None."""

    orb_start_min = int(params["orb_start_utc_hour"]) * 60 + int(FIXED["orb_start_utc_min"])
    orb_end_min   = orb_start_min + int(params["orb_duration_min"])
    rr            = params["tp_rr_ratio"]
    delta_th      = params["delta_threshold"]
    close_min     = int(FIXED["trade_end_utc_hour"]) * 60

    trades = []

    for day in days:
        dm = day["minutes"]

        orb_mask  = (dm >= orb_start_min) & (dm < orb_end_min)
        trig_mask = (dm >= orb_end_min)   & (dm < close_min)

        if not orb_mask.any() or not trig_mask.any():
            continue

        orb_high = day["high"][orb_mask].max()
        orb_low  = day["low"][orb_mask].min()

        closes  = day["close"][trig_mask]
        highs   = day["high"][trig_mask]
        lows    = day["low"][trig_mask]
        spreads = day["spread"][trig_mask]

        # vectorized entry detection
        entry_mask = closes > orb_high
        if delta_th > 0 and day["delta"] is not None:
            entry_mask = entry_mask & (day["delta"][trig_mask] >= delta_th)

        if not entry_mask.any():
            continue

        ei     = int(np.argmax(entry_mask))
        entry  = closes[ei] + spreads[ei] / 2
        rng    = entry - orb_low
        if rng <= 0:
            continue

        tp_price = entry + rr * rng
        sl_price = orb_low

        post_h = highs[ei + 1:]
        post_l = lows[ei + 1:]

        tp_hits = np.nonzero(post_h >= tp_price)[0]
        sl_hits = np.nonzero(post_l <= sl_price)[0]

        tp_i = int(tp_hits[0]) if len(tp_hits) else len(post_h)
        sl_i = int(sl_hits[0]) if len(sl_hits) else len(post_h)

        if tp_i <= sl_i and tp_i < len(post_h):
            pnl = tp_price - entry
        elif sl_i < tp_i and sl_i < len(post_h):
            pnl = sl_price - entry
        else:
            pnl = closes[-1] - entry

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
        "_trades":     trades,   # lista PnL individuali per Monte Carlo
    }


def run():
    # ── Carica M5 ─────────────────────────────────────────────────────────────
    if not M5_PATH.exists():
        print(f"M5 file non trovato: {M5_PATH}")
        print("Esegui prima: python src/pipeline.py")
        return

    print(f"Carico M5 in-sample ({IS_START} -> {IS_END})...")
    m5 = pd.read_parquet(M5_PATH)
    m5 = m5[(m5.index >= IS_START) & (m5.index <= IS_END)]
    print(f"  {len(m5):,} barre M5 caricate")

    # ── Pre-split dati ───────────────────────────────────────────────────────
    print("Pre-processing dati giornalieri...")
    days = prepare_daily_data(m5)
    print(f"  {len(days)} giorni pronti")

    # ── Genera griglia ────────────────────────────────────────────────────────
    grid = make_grid(OPTIMIZE)
    print(f"\nCombinazioni totali: {len(grid):,}")
    print("Avvio ottimizzazione...\n")

    # ── Esegui ────────────────────────────────────────────────────────────────
    results = []
    t0 = datetime.now()

    iterator = tqdm(grid, unit="combo", ncols=80) if HAS_TQDM else grid

    try:
        for i, params in enumerate(iterator):
            res = vectorized_backtest(days, params)
            if res:
                row = {**params, **res}
                results.append(row)

            if not HAS_TQDM and (i + 1) % max(1, len(grid) // 20) == 0:
                pct   = (i + 1) / len(grid) * 100
                ela   = (datetime.now() - t0).seconds
                eta_s = ela / (i + 1) * (len(grid) - i - 1)
                print(f"  {pct:.0f}%  [{i+1}/{len(grid)}]  ETA {eta_s/60:.1f} min", end="\r")

    except KeyboardInterrupt:
        print("\n\nInterrotto. Salvo risultati parziali...")

    elapsed_min = (datetime.now() - t0).seconds / 60
    print(f"\nCompletato in {elapsed_min:.1f} min")
    print(f"Combinazioni con trade: {len(results):,} / {len(grid):,}")

    # ── Salva risultati ───────────────────────────────────────────────────────
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results).sort_values("net_profit", ascending=False)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nRisultati -> {OUT_CSV}")

    # ── Top 20 ───────────────────────────────────────────────────────────────
    print("\n── TOP 20 per net_profit ──────────────────────────────────────────")
    cols = ["orb_start_utc_hour", "orb_duration_min",
            "tp_rr_ratio", "delta_threshold",
            "net_profit", "n_trades", "win_rate", "avg_trade", "max_dd"]
    print(df[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    run()
