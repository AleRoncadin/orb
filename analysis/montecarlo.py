"""
montecarlo.py — Shuffle-trades Monte Carlo (metodo Matteo Conti)

Input:  backtests/results/trades_<label>.csv
Output: plot equity curves + statistiche robustezza

Domande a cui risponde:
1. Edge è sequence-dependent? (se sì → fragile)
2. Expected value per trade con confidence interval
3. Worst-case drawdown su 1000 simulazioni
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "backtests" / "results"
PLOTS_DIR   = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

N_SIMULATIONS = 1_000
RANDOM_SEED   = 42


def load_pnls(label: str) -> np.ndarray:
    path = RESULTS_DIR / f"trades_{label}.csv"
    df   = pd.read_csv(path)
    return df["realized_pnl"].astype(float).dropna().values


def run_montecarlo(pnls: np.ndarray, n_sim: int = N_SIMULATIONS) -> dict:
    rng = np.random.default_rng(RANDOM_SEED)
    n   = len(pnls)

    final_equities = []
    max_drawdowns  = []
    equity_curves  = []

    for _ in range(n_sim):
        shuffled = rng.permutation(pnls)
        equity   = np.cumsum(shuffled)
        curve    = np.concatenate([[0], equity])

        # max drawdown
        peak = np.maximum.accumulate(curve)
        dd   = np.max(peak - curve)

        final_equities.append(curve[-1])
        max_drawdowns.append(dd)
        equity_curves.append(curve)

    ev_per_trade = np.array(final_equities) / n

    return {
        "equity_curves":  np.array(equity_curves),
        "final_equities": np.array(final_equities),
        "max_drawdowns":  np.array(max_drawdowns),
        "ev_per_trade":   ev_per_trade,
        "n_trades":       n,
    }


def print_stats(res: dict, label: str) -> None:
    ev = res["ev_per_trade"]
    dd = res["max_drawdowns"]
    fe = res["final_equities"]

    print(f"\n{'='*50}")
    print(f"Monte Carlo — {label}  ({N_SIMULATIONS} simulazioni)")
    print(f"Trades per simulazione : {res['n_trades']}")
    print(f"{'─'*50}")
    print(f"Expected value/trade   : ${np.mean(ev):.2f}")
    print(f"  95% CI               : [${np.percentile(ev,2.5):.2f}, ${np.percentile(ev,97.5):.2f}]")
    print(f"Net profit  median     : ${np.median(fe):.0f}")
    print(f"Net profit  5th pct    : ${np.percentile(fe,5):.0f}  (worst 5%)")
    print(f"Max drawdown median    : ${np.median(dd):.0f}")
    print(f"Max drawdown 95th pct  : ${np.percentile(dd,95):.0f}  (worst 5%)")
    pct_profitable = np.mean(fe > 0) * 100
    print(f"% simulazioni profit.  : {pct_profitable:.1f}%")
    print(f"{'='*50}")


def plot(res: dict, label: str) -> None:
    curves = res["equity_curves"]
    n_trades = res["n_trades"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Monte Carlo — {label}", fontsize=13)

    # Left: equity curves
    ax = axes[0]
    for c in curves[:200]:
        ax.plot(c, alpha=0.05, color="steelblue", linewidth=0.5)
    ax.plot(np.median(curves, axis=0), color="orange", linewidth=1.5, label="median")
    ax.plot(np.percentile(curves, 5,  axis=0), color="red",   linewidth=1, linestyle="--", label="5th pct")
    ax.plot(np.percentile(curves, 95, axis=0), color="green", linewidth=1, linestyle="--", label="95th pct")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Cumulative PnL ($)")
    ax.set_title("Shuffle equity curves (200 shown)")
    ax.legend()

    # Right: final equity distribution
    ax2 = axes[1]
    fe = res["final_equities"]
    bins = 50 if fe.max() - fe.min() > 1e-6 else 1
    ax2.hist(fe, bins=bins, color="steelblue", edgecolor="white")
    ax2.axvline(0, color="red", linewidth=1.5, label="breakeven")
    ax2.axvline(np.median(res["final_equities"]), color="orange", linewidth=1.5, label="median")
    ax2.set_xlabel("Final equity ($)")
    ax2.set_title("Final equity distribution")
    ax2.legend()

    plt.tight_layout()
    out = PLOTS_DIR / f"montecarlo_{label}.png"
    plt.savefig(out, dpi=150)
    print(f"Plot saved: {out}")
    plt.close()


def analyze(label: str = "baseline") -> None:
    pnls = load_pnls(label)
    print(f"Loaded {len(pnls)} trades from '{label}'")
    res = run_montecarlo(pnls)
    print_stats(res, label)
    plot(res, label)


if __name__ == "__main__":
    analyze("baseline")
