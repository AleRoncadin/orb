"""
strategy_orb.py — ORB fedele al modello Fabio Valentini (validato Matteo Conti)

Logica originale:
  - LONG ONLY
  - ORB window: parametrizzabile (default 08:30–09:00 ET → 13:30–14:00 UTC)
  - Entry: Close > ORB_High + BarDelta >= delta_threshold
  - SL: ORB_Low (dinamico, non fisso)
  - TP: entry + tp_rr_ratio * (entry - ORB_Low)  [range-width scaled]
  - EOD: chiusura forzata a trade_end_time
  - 1 trade per sessione

Delta filter su XAUUSD: approssimato da tick direction
  - Last Price == Ask → buy tick → delta +1
  - Last Price == Bid → sell tick → delta -1
  Pipeline pre-calcola bar_delta su M5 e lo salva nel parquet.
"""

from decimal import Decimal
from typing import Optional
import pandas as pd

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.trading.strategy import Strategy


class ORBConfig(StrategyConfig, frozen=True):
    instrument_id: str        = "XAUUSD.DUKASCOPY"
    bar_type: str             = "XAUUSD.DUKASCOPY-5-MINUTE-BID-INTERNAL"

    # ORB window (UTC) — ottimizzabili
    orb_start_utc_hour: int   = 13      # 13:30 UTC = 08:30 ET (winter)
    orb_start_utc_min:  int   = 30
    orb_duration_min:   int   = 30      # 30 min → finestra 08:30–09:00

    # Exit
    tp_rr_ratio: float        = 1.0     # TP = entry + ratio * (entry - ORB_Low)
    trade_end_utc_hour: int   = 19      # 19:00 UTC = 14:00 ET (winter)
    trade_end_utc_min:  int   = 0

    # Delta filter (approssimato da ticks)
    delta_threshold: int      = 200     # BarDelta minimo per entrare
    use_delta_filter: bool    = True    # False → disabilita filtro

    trade_size: float         = 1.0


class ORBStrategy(Strategy):

    def __init__(self, config: ORBConfig) -> None:
        super().__init__(config)
        self.cfg = config

        self._orb_high: Optional[float] = None
        self._orb_low:  Optional[float] = None
        self._orb_built: bool = False
        self._traded_today: bool = False
        self._last_date = None
        self._bar_delta: float = 0.0    # delta corrente della bar

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def on_start(self) -> None:
        self._instrument = self.cache.instrument(
            InstrumentId.from_str(self.cfg.instrument_id)
        )
        self.subscribe_bars(BarType.from_str(self.cfg.bar_type))

    def on_stop(self) -> None:
        iid = InstrumentId.from_str(self.cfg.instrument_id)
        self.close_all_positions(iid)
        self.cancel_all_orders(iid)

    # ── Bar handler ──────────────────────────────────────────────────────────

    def on_bar(self, bar: Bar) -> None:
        bar_time = pd.Timestamp(bar.ts_event, unit="ns", tz="UTC")
        bar_date = bar_time.date()

        # Daily reset
        if bar_date != self._last_date:
            self._reset_daily()
            self._last_date = bar_date

        bar_min       = bar_time.hour * 60 + bar_time.minute
        orb_start_min = self.cfg.orb_start_utc_hour * 60 + self.cfg.orb_start_utc_min
        orb_end_min   = orb_start_min + self.cfg.orb_duration_min
        trade_end_min = self.cfg.trade_end_utc_hour * 60 + self.cfg.trade_end_utc_min

        # EOD force-close
        if bar_min >= trade_end_min:
            self._force_close()
            return

        # Phase 1: accumulate ORB range
        if orb_start_min <= bar_min < orb_end_min:
            h = float(bar.high)
            l = float(bar.low)
            self._orb_high = max(self._orb_high, h) if self._orb_high else h
            self._orb_low  = min(self._orb_low,  l) if self._orb_low  else l
            return

        # Mark ORB as built on first bar after window
        if bar_min >= orb_end_min and not self._orb_built:
            self._orb_built = True

        # Phase 2: entry logic
        if (self._orb_built
                and not self._traded_today
                and self._orb_high is not None
                and self._orb_low  is not None
                and bar_min < trade_end_min):
            self._check_entry(bar)

    # ── Entry ────────────────────────────────────────────────────────────────

    def _check_entry(self, bar: Bar) -> None:
        close = float(bar.close)

        # Long only: close must break above ORB_High
        if close <= self._orb_high:
            return

        # Delta filter (approssimato — pipeline deve fornire bar_delta)
        if self.cfg.use_delta_filter:
            if self._bar_delta < self.cfg.delta_threshold:
                return

        # SL = ORB_Low, TP = entry + RR * range
        entry    = close
        sl_price = self._orb_low
        rng      = entry - sl_price

        if rng <= 0:
            return

        tp_price = entry + self.cfg.tp_rr_ratio * rng

        self._enter_long(entry, sl_price, tp_price)
        self._traded_today = True

    def _enter_long(self, entry: float, sl: float, tp: float) -> None:
        iid = InstrumentId.from_str(self.cfg.instrument_id)
        qty = Quantity.from_str(str(self.cfg.trade_size))

        self.submit_order(
            self.order_factory.market(
                instrument_id=iid,
                order_side=OrderSide.BUY,
                quantity=qty,
                time_in_force=TimeInForce.GTC,
            )
        )
        self.submit_order(
            self.order_factory.limit(
                instrument_id=iid,
                order_side=OrderSide.SELL,
                quantity=qty,
                price=Price(tp, precision=2),
                time_in_force=TimeInForce.GTC,
                reduce_only=True,
            )
        )
        self.submit_order(
            self.order_factory.stop_market(
                instrument_id=iid,
                order_side=OrderSide.SELL,
                quantity=qty,
                trigger_price=Price(sl, precision=2),
                time_in_force=TimeInForce.GTC,
                reduce_only=True,
            )
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _reset_daily(self) -> None:
        self._orb_high    = None
        self._orb_low     = None
        self._orb_built   = False
        self._traded_today = False
        self._bar_delta   = 0.0

    def _force_close(self) -> None:
        iid = InstrumentId.from_str(self.cfg.instrument_id)
        for pos in self.cache.positions_open(instrument_id=iid):
            self.close_position(pos)
        self.cancel_all_orders(iid)
