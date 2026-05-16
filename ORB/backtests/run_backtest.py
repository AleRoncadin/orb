"""
run_backtest.py — In-sample backtest XAUUSD ORB

In-sample : 2015-01-01 → 2022-12-31
Out-sample: 2023-01-01 → 2025-12-31  (NON toccare durante ottimizzazione)

Slippage: FillModel probabilistico (0.5 pip medio XAUUSD)
Spread:   già nei QuoteTick (bid/ask reali da Tickstory)
"""

from decimal import Decimal
from pathlib import Path
import pandas as pd

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.backtest.models import FillModel
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import TraderId, Venue
from nautilus_trader.model.objects import Money
from nautilus_trader.persistence.catalog import ParquetDataCatalog

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from strategy_orb import ORBConfig, ORBStrategy

CATALOG_PATH = Path(__file__).parent.parent / "data" / "catalog"
RESULTS_DIR  = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Periodo ──────────────────────────────────────────────────────────────────
IS_START = pd.Timestamp("2015-01-01", tz="UTC")
IS_END   = pd.Timestamp("2022-12-31 23:59:59", tz="UTC")

# ── Slippage model ───────────────────────────────────────────────────────────
# XAUUSD slippage realistico: ~0.5–1 pip su market order
FILL_MODEL = FillModel(
    prob_fill_on_limit=1.0,
    prob_fill_on_stop=0.95,
    prob_slippage=0.4,      # 40% chance di 1-tick extra slippage
    random_seed=42,
)


def run_single(config: ORBConfig, label: str = "default") -> dict:
    catalog = ParquetDataCatalog(str(CATALOG_PATH))

    instrument = catalog.instruments(instrument_ids=["XAUUSD.DUKASCOPY"])[0]
    ticks = catalog.quote_ticks(
        instrument_ids=["XAUUSD.DUKASCOPY"],
        start=IS_START,
        end=IS_END,
    )

    engine_cfg = BacktestEngineConfig(trader_id=TraderId("BACKTESTER-001"))
    engine = BacktestEngine(config=engine_cfg)

    engine.add_venue(
        venue=Venue("DUKASCOPY"),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USD,
        starting_balances=[Money(100_000, USD)],
        fill_model=FILL_MODEL,
    )

    engine.add_instrument(instrument)
    engine.add_data(ticks)
    engine.add_strategy(ORBStrategy(config=config))

    engine.run()

    account = engine.trader.generate_account_report(Venue("DUKASCOPY"))
    trades  = engine.trader.generate_order_fills_report()

    # ── Metriche chiave ───────────────────────────────────────────────────
    pnls = trades["realized_pnl"].astype(float).dropna().tolist()
    net_profit   = sum(pnls)
    n_trades     = len(pnls)
    win_rate     = sum(1 for p in pnls if p > 0) / n_trades if n_trades else 0
    avg_trade    = net_profit / n_trades if n_trades else 0
    max_dd       = _max_drawdown(pnls)

    results = {
        "label":       label,
        "net_profit":  round(net_profit, 2),
        "n_trades":    n_trades,
        "win_rate":    round(win_rate, 4),
        "avg_trade":   round(avg_trade, 2),
        "max_dd":      round(max_dd, 2),
    }
    print(results)

    trades.to_csv(RESULTS_DIR / f"trades_{label}.csv", index=False)
    engine.dispose()
    return results


def _max_drawdown(pnls: list[float]) -> float:
    equity = 0.0
    peak   = 0.0
    max_dd = 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    return max_dd


if __name__ == "__main__":
    cfg = ORBConfig(
        orb_start_utc_hour=13,
        orb_start_utc_min=30,
        orb_duration_min=15,
        take_profit_usd=4.50,
        stop_loss_usd=1.00,
    )
    run_single(cfg, label="baseline")
