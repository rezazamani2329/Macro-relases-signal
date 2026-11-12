"""Alpha of the surprise strategies relative to being long in the same windows (Table: tab_alpha).

Run from the repo root:  python scripts/alpha_regression.py
Uses the walk-forward trade logs in results/ (2019-2026, base cost 2 ticks + $4.50).
For each day: net strategy return (sum of trades) regressed on the long return in the same windows,
r_net = alpha + beta * r_long + e, with HC1 (heteroskedasticity-robust) t-statistics.
"""
import numpy as np
import pandas as pd

def ols_hc1(y, x):
    X = np.column_stack([np.ones(len(x)), x])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    n = len(y)
    XtX = np.linalg.inv(X.T @ X)
    V = XtX @ (X.T * e**2) @ X @ XtX * n / (n - 2)
    return b, b / np.sqrt(np.diag(V)), 1 - e.var() / y.var(), n

MARKETS = [("ES", "nb04_", "nb05_"), ("NQ", "nb04b_NQ_", "nb05b_NQ_"),
           ("YM", "nb04b_YM_", "nb05b_YM_"), ("ZN", "nb04b_ZN_", "nb05b_ZN_")]
rows = []
for m, p4, p5 in MARKETS:
    for pref, strat, ev, label in [(p4, "Naive sign", "CPI", "CPI naive"),
                                   (p4, "Surprise only", "CPI", "CPI model"),
                                   (p5, "All · naive", None, "Broad naive")]:
        d = pd.read_csv(f"results/{pref}trade_log.csv")
        d = d[d.strategy == strat]
        if ev:
            d = d[d.event_type == ev]
        d["ts"] = pd.to_datetime(d.timestamp_utc)
        d = d[(d.ts >= "2019-01-01") & (d.position != 0)]
        if len(d) < 10:
            continue
        g = d.groupby(d.ts.dt.date).agg(net=("net_bps", "sum"), long=("ret_bps", "sum"))
        b, t, r2, n = ols_hc1(g.net.values, g.long.values)
        rows.append(dict(market=m, strategy=label, trades=len(d), days=n, mean_net_bps=d.net_bps.mean(),
                         alpha_bps=b[0], t_alpha=t[0], beta=b[1], t_beta=t[1], r2=r2))
out = pd.DataFrame(rows)
print(out.round(2).to_string(index=False))
out.to_csv("results/alpha_regression.csv", index=False)
