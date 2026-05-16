"""
pipeline.py — Tickstory CSV → NautilusTrader ParquetDataCatalog

Input CSV format (Tickstory UTC+2):
    Date,Timestamp,Bid Price,Ask Price,Last Price,Volume
    20150102,01:02:10,1248.23,1248.71,1248.23,1

Output: data/catalog/  (QuoteTick parquet, ready for BacktestEngine)
"""

import pandas as pd
from pathlib import Path
from decimal import Decimal
from datetime import timezone, timedelta

from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import CurrencyPair
from nautilus_trader.model.currencies import Currency
from nautilus_trader.persistence.catalog import ParquetDataCatalog

RAW_CSV   = Path(__file__).parent.parent / "data" / "raw" / "XAUUSD.csv"
CATALOG   = Path(__file__).parent.parent / "data" / "catalog"
CHUNK_SZ  = 500_000          # rows per write chunk (RAM control)
UTC_PLUS2 = timezone(timedelta(hours=0))  # UTC puro — seleziona UTC in Tickstory

# ── Instrument definition ────────────────────────────────────────────────────

VENUE = Venue("DUKASCOPY")

def make_instrument() -> CurrencyPair:
    XAU = Currency.from_str("XAU")
    USD = Currency.from_str("USD")
    return CurrencyPair(
        instrument_id=InstrumentId(Symbol("XAUUSD"), VENUE),
        raw_symbol=Symbol("XAUUSD"),
        base_currency=XAU,
        quote_currency=USD,
        price_precision=2,
        size_precision=2,
        price_increment=Price.from_str("0.01"),
        size_increment=Quantity.from_str("0.01"),
        lot_size=Quantity.from_str("1.0"),
        max_quantity=None,
        min_quantity=Quantity.from_str("0.01"),
        max_notional=None,
        min_notional=None,
        max_price=None,
        min_price=None,
        margin_init=Decimal("0.01"),
        margin_maint=Decimal("0.005"),
        maker_fee=Decimal("0.0"),
        taker_fee=Decimal("0.0"),
        ts_event=0,
        ts_init=0,
    )


# ── CSV parsing ──────────────────────────────────────────────────────────────

def parse_chunk(df: pd.DataFrame, instrument_id: InstrumentId) -> list[QuoteTick]:
    """Convert a raw DataFrame chunk to QuoteTick list."""
    # Combine date + time → datetime UTC
    dt = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Timestamp"],
        format="%Y%m%d %H:%M:%S",
    ).dt.tz_localize("UTC")

    ts_ns = dt.astype("int64").values   # nanoseconds since epoch

    ticks = []
    for i in range(len(df)):
        bid = df["Bid Price"].iloc[i]
        ask = df["Ask Price"].iloc[i]
        if bid <= 0 or ask <= 0 or ask < bid:
            continue                    # skip corrupt rows
        ticks.append(
            QuoteTick(
                instrument_id=instrument_id,
                bid_price=Price(bid, precision=2),
                ask_price=Price(ask, precision=2),
                bid_size=Quantity(1.0, precision=2),
                ask_size=Quantity(1.0, precision=2),
                ts_event=int(ts_ns[i]),
                ts_init=int(ts_ns[i]),
            )
        )
    return ticks


# ── Main ─────────────────────────────────────────────────────────────────────

def run():
    import shutil

    if not RAW_CSV.exists():
        raise FileNotFoundError(f"CSV not found: {RAW_CSV}\nRun Tickstory first.")

    # wipe catalog to avoid non-disjoint interval errors on re-run
    if CATALOG.exists():
        shutil.rmtree(CATALOG)
    CATALOG.mkdir(parents=True, exist_ok=True)
    catalog = ParquetDataCatalog(str(CATALOG))

    instrument = make_instrument()
    catalog.write_data([instrument])
    print(f"Instrument written: {instrument.id}")

    # Read all CSV into DataFrame, sort, then write monthly to catalog.
    # Monthly write = no overlap at chunk boundaries, low RAM peak per batch.
    print("Reading CSV...")
    reader = pd.read_csv(RAW_CSV, chunksize=CHUNK_SZ, dtype={"Date": str})
    chunks = []
    for i, chunk in enumerate(reader):
        dt = pd.to_datetime(
            chunk["Date"].astype(str) + " " + chunk["Timestamp"],
            format="%Y%m%d %H:%M:%S",
        ).dt.tz_localize("UTC")
        chunk.index = dt
        chunks.append(chunk)
        print(f"  Chunk {i+1} read: {len(chunk):,} rows")

    df = pd.concat(chunks).sort_index()
    print(f"Total rows: {len(df):,}  Range: {df.index[0]} -> {df.index[-1]}")

    # write month by month — each month is sorted and non-overlapping
    total = 0
    for period, month_df in df.groupby(df.index.to_period("M")):
        ticks = parse_chunk(month_df, instrument.id)
        if ticks:
            catalog.write_data(ticks)
            total += len(ticks)
        print(f"  {period}: {len(ticks):,} ticks written (total: {total:,})")

    print(f"\nDone. {total:,} ticks in catalog -> {CATALOG}")


def build_m5(csv_path: Path = RAW_CSV) -> None:
    """Costruisce M5 OHLCV + spread medio + bar_delta da CSV tick. Salva parquet."""
    m5_path = csv_path.parent.parent / "m5_xauusd.parquet"
    print("Costruisco M5 da tick CSV...")

    chunks = []
    reader = pd.read_csv(csv_path, chunksize=500_000, dtype={"Date": str})
    for chunk in reader:
        dt = pd.to_datetime(
            chunk["Date"].astype(str) + " " + chunk["Timestamp"],
            format="%Y%m%d %H:%M:%S",
        ).dt.tz_localize("UTC")
        chunk.index = dt
        chunk["mid"]      = (chunk["Bid Price"] + chunk["Ask Price"]) / 2
        chunk["spread"]   = chunk["Ask Price"] - chunk["Bid Price"]
        # buy tick if Last >= Ask, sell tick if Last <= Bid
        chunk["tick_dir"] = 0
        chunk.loc[chunk["Last Price"] >= chunk["Ask Price"], "tick_dir"] = 1
        chunk.loc[chunk["Last Price"] <= chunk["Bid Price"], "tick_dir"] = -1
        chunks.append(chunk[["mid", "spread", "tick_dir"]])

    df = pd.concat(chunks).sort_index()

    m5 = df["mid"].resample("5min").ohlc()
    m5["spread"]    = df["spread"].resample("5min").mean()
    m5["bar_delta"] = df["tick_dir"].resample("5min").sum()
    m5 = m5.dropna()

    m5.to_parquet(m5_path)
    print(f"M5 salvato: {m5_path}  ({len(m5):,} barre)")


if __name__ == "__main__":
    from decimal import Decimal
    run()          # catalog per NautilusTrader
    build_m5()     # M5 parquet per optimizer
