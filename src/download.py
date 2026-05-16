"""
Dukascopy tick data downloader.
Output: data/raw/XAUUSD.csv  (timestamp_utc, bid, ask, bid_vol, ask_vol)
"""

from duka.app import app
from datetime import date
from pathlib import Path
import time

INSTRUMENTS = ["XAUUSD"]
START       = date(2015, 1, 1)
END         = date(2025, 12, 31)
WORKERS     = 4            # parallel HTTP workers (16 causa freeze su Windows)
TIMEFRAME   = "tick"
OUT_DIR     = Path(__file__).parent.parent / "data" / "raw"

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Download  : {INSTRUMENTS}")
    print(f"Period    : {START} → {END}")
    print(f"Workers   : {WORKERS}")
    print(f"Output    : {OUT_DIR}")
    print("-" * 40)
    t0 = time.time()
    app(INSTRUMENTS, START, END, WORKERS, TIMEFRAME, str(OUT_DIR), True)
    elapsed = (time.time() - t0) / 60
    print(f"\nDone in {elapsed:.1f} min")
