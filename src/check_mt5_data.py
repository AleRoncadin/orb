import MetaTrader5 as mt5
from datetime import datetime

if not mt5.initialize():
    print(f"MT5 init failed: {mt5.last_error()}")
    quit()

print(f"MT5 connected: {mt5.terminal_info().name}")
print()

symbol = "XAUUSD"
info = mt5.symbol_info(symbol)
if info is None:
    print(f"{symbol} not found. Available symbols with XAU:")
    symbols = mt5.symbols_get()
    for s in symbols:
        if "XAU" in s.name or "GOLD" in s.name.upper():
            print(f"  {s.name}")
    mt5.shutdown()
    quit()

print(f"Symbol    : {info.name}")
print(f"Spread    : {info.spread} points")
print(f"Digits    : {info.digits}")
print()

# check oldest available tick
ticks = mt5.copy_ticks_from(symbol, datetime(2015, 1, 1), 1, mt5.COPY_TICKS_ALL)
if ticks is not None and len(ticks) > 0:
    oldest = datetime.utcfromtimestamp(ticks[0]['time'])
    print(f"Oldest tick available : {oldest}")
else:
    print("No ticks from 2015 — trying 2020...")
    ticks = mt5.copy_ticks_from(symbol, datetime(2020, 1, 1), 1, mt5.COPY_TICKS_ALL)
    if ticks is not None and len(ticks) > 0:
        oldest = datetime.utcfromtimestamp(ticks[0]['time'])
        print(f"Oldest tick available : {oldest}")
    else:
        print("No ticks found even from 2020")

# check M1 bars availability
rates = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_M1, datetime(2015, 1, 1), 1)
if rates is not None and len(rates) > 0:
    oldest_m1 = datetime.utcfromtimestamp(rates[0]['time'])
    print(f"Oldest M1 bar available: {oldest_m1}")

mt5.shutdown()
