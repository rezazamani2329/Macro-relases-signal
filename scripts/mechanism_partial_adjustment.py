"""What drives the post-release drift? Partial adjustment versus catch-up to bonds (paper Table: tab_mechanism).

Run from the repo root:  python scripts/mechanism_partial_adjustment.py
Needs the licensed event files in data/processed (not in the repository, see .gitignore).
Output: results/mechanism_partial_adjustment.csv.

For CPI, NFP and PCE releases, 2016-2026, with release-family fixed effects and HC1 t-statistics:
  (1) equity drift D (minute 1 to 30, bps) on the standardized surprise z;
  (2) D on z, the market's own first-minute return J and the first-minute return of 10-year
      Treasury futures J_ZN. If equities were catching up with the rate information that the bond
      market prices first, J_ZN would predict D. If the drift is a partial adjustment of the equity
      market's own reaction, J would predict D and absorb z;
  (3) the coefficient on J in subsamples (tightening cycle in/out, excluding 2022, excluding 2020,
      returns winsorized at 1%/99%);
  (4) the reverse test: ZN drift on its own first-minute return and on the equity first-minute return;
  (5) as a direction signal, 2019-2026 CPI releases: mean of sign(J)*D against sign(z)*D (bps, before costs).
"""
import numpy as np
import pandas as pd

MARKETS = {"ES": "", "NQ": "_NQ", "YM": "_YM", "ZN": "_ZN"}
HIKES = [("2015-12-16", "2018-12-19"), ("2022-03-16", "2023-07-26")]


def ols_hc1(y, X):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    n, k = X.shape
    XtX = np.linalg.inv(X.T @ X)
    V = XtX @ (X.T * e**2) @ X @ XtX * n / (n - k)
    return b, b / np.sqrt(np.diag(V))


def load(m):
    d = pd.read_csv(f"data/processed/event_state_features_2016_2026{MARKETS[m]}.csv")
    d = d.dropna(subset=["x", "drift_30m_bps", "ret_1m_bps"]).copy()
    d["z"] = d.x / d.groupby("event_type").x.transform("std")
    return d


def reg(s, y, cols):
    fam = pd.get_dummies(s.event_type, drop_first=True).astype(float).values
    X = np.column_stack([np.ones(len(s))] + [s[c].values for c in cols] + [fam])
    b, t = ols_hc1(s[y].values, X)
    return {c: (b[i + 1], t[i + 1]) for i, c in enumerate(cols)}


def main():
    zn = load("ZN")[["timestamp_utc", "ret_1m_bps", "drift_30m_bps"]]
    zn = zn.rename(columns={"ret_1m_bps": "J_ZN", "drift_30m_bps": "D_ZN"})
    rows = []

    def add(m, spec, sample, res, n):
        for var, (b, t) in res.items():
            rows.append(dict(market=m, spec=spec, sample=sample, n=n, variable=var, coef=b, t=t))

    for m in ["ES", "NQ", "YM"]:
        d = load(m).merge(zn, on="timestamp_utc").dropna(subset=["J_ZN"]).rename(columns={"ret_1m_bps": "J"})
        d["date"] = pd.to_datetime(d.date_et)
        d["HIKE"] = 0
        for a, b in HIKES:
            d.loc[(d.date >= a) & (d.date <= b), "HIKE"] = 1
        w = d.copy()
        for c in ["J", "drift_30m_bps", "J_ZN", "D_ZN"]:
            lo, hi = w[c].quantile([0.01, 0.99])
            w[c] = w[c].clip(lo, hi)
        add(m, "(1) D on z", "All", reg(d, "drift_30m_bps", ["z"]), len(d))
        add(m, "(2) D on z, J, J_ZN", "All", reg(d, "drift_30m_bps", ["z", "J", "J_ZN"]), len(d))
        add(m, "(2) D on z, J, J_ZN", "CPI", reg(d[d.event_type == "CPI"], "drift_30m_bps", ["z", "J", "J_ZN"]),
            int((d.event_type == "CPI").sum()))
        for name, s in [("Tightening cycle", d[d.HIKE == 1]), ("Outside tightening", d[d.HIKE == 0]),
                        ("Excluding 2022", d[d.date.dt.year != 2022]), ("Excluding 2020", d[d.date.dt.year != 2020]),
                        ("Winsorized 1%", w), ("2016-2018", d[d.date.dt.year <= 2018])]:
            add(m, "(3) D on J, J_ZN", name, reg(s, "drift_30m_bps", ["J", "J_ZN"]), len(s))
        add(m, "(4) D_ZN on J_ZN, J", "All", reg(d, "D_ZN", ["J_ZN", "J"]), len(d))
        add(m, "(4) D_ZN on J_ZN, J", "Winsorized 1%", reg(w, "D_ZN", ["J_ZN", "J"]), len(w))
        c = d[(d.event_type == "CPI") & (d.date.dt.year >= 2019)]
        for var, sig in [("sign(J)", np.sign(c.J)), ("sign(z)", np.sign(c.z))]:
            r = sig * c.drift_30m_bps
            rows.append(dict(market=m, spec="(5) mean signed drift, CPI 2019-2026", sample="CPI", n=len(r),
                             variable=var, coef=r.mean(), t=r.mean() / r.std() * np.sqrt(len(r))))
    out = pd.DataFrame(rows)
    print(out.round(3).to_string(index=False))
    out.to_csv("results/mechanism_partial_adjustment.csv", index=False)


if __name__ == "__main__":
    main()
