"""Does the post-release drift depend on the monetary-policy regime? (paper Table: tab_regime)

Run from the repo root:  python scripts/regime_drift.py
Needs the licensed inputs in data/raw and data/processed, which are not in the repository
(see .gitignore); the output is saved in results/regime_drift.csv.

For each market we regress the drift from minute 1 to minute 30 after a release on the
standardized consensus surprise and its interaction with a regime indicator known before the release:
    drift = a + b*z + c*(z x R) + d*R + family fixed effects + e
with HC1 (heteroskedasticity-robust) t-statistics. z is the surprise divided by its standard
deviation within each release family (CPI, NFP, PCE), signed so that a positive z predicts a
positive first-minute return in that market.

Regimes:
  HIKE   - the Federal Reserve is in a tightening cycle (first to last hike):
           2015-12-16 to 2018-12-19 and 2022-03-16 to 2023-07-26 (public FOMC decisions).
  HIGHINF- the most recent CPI year-over-year rate published before the release exceeds 3%
           (from the Bloomberg release calendar, ticker CPI YOY Index).
Each specification is estimated on all releases (2016-2026) and on all releases outside 2022.
"""
import numpy as np
import pandas as pd

HIKES = [("2015-12-16", "2018-12-19"), ("2022-03-16", "2023-07-26")]
MARKETS = {"ES": "", "NQ": "_NQ", "YM": "_YM", "ZN": "_ZN"}


def cpi_yoy_history(path="data/raw/Bloomberg Economic Releases.xlsx"):
    rows = []
    for sh in pd.ExcelFile(path).sheet_names:
        d = pd.read_excel(path, sheet_name=sh, header=None, skiprows=1)
        d = d[d[11].astype(str).str.strip() == "CPI YOY Index"]
        rows.append(pd.DataFrame({"date": pd.to_datetime(d[2]), "cpi_yoy": pd.to_numeric(d[7], errors="coerce")}))
    return pd.concat(rows).dropna().drop_duplicates("date").sort_values("date").reset_index(drop=True)


def ols_hc1(y, X):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    n, k = X.shape
    XtX = np.linalg.inv(X.T @ X)
    V = XtX @ (X.T * e**2) @ X @ XtX * n / (n - k)
    return b, b / np.sqrt(np.diag(V))


def main():
    cpi = cpi_yoy_history()
    out = []
    for m, suf in MARKETS.items():
        d = pd.read_csv(f"data/processed/event_state_features_2016_2026{suf}.csv")
        d = d.dropna(subset=["x", "drift_30m_bps"]).copy()
        d["date"] = pd.to_datetime(d.date_et)
        d["z"] = d.x / d.groupby("event_type").x.transform("std")
        d["HIKE"] = 0
        for a, b in HIKES:
            d.loc[(d.date >= a) & (d.date <= b), "HIKE"] = 1
        # last CPI YoY published strictly before the release date (no look-ahead)
        d = d.sort_values("date")
        prev = cpi.rename(columns={"date": "cpi_date"})
        prev["cpi_date"] = prev.cpi_date + pd.Timedelta(days=1)
        d = pd.merge_asof(d, prev, left_on="date", right_on="cpi_date", direction="backward")
        d["HIGHINF"] = (d.cpi_yoy > 3.0).astype(int)
        for reg in ["HIKE", "HIGHINF"]:
            for sample, s in [("All years", d), ("Excluding 2022", d[d.date.dt.year != 2022])]:
                fam = pd.get_dummies(s.event_type, drop_first=True).astype(float).values
                X = np.column_stack([np.ones(len(s)), s.z, s.z * s[reg], s[reg], fam])
                b, t = ols_hc1(s.drift_30m_bps.values, X)
                out.append(dict(market=m, regime=reg, sample=sample, n=len(s), n_regime=int(s[reg].sum()),
                                b_surprise=b[1], t_surprise=t[1], c_interaction=b[2], t_interaction=t[2],
                                drift_in_regime=b[1] + b[2]))
    res = pd.DataFrame(out)
    print(res.round(2).to_string(index=False))
    res.to_csv("results/regime_drift.csv", index=False)


if __name__ == "__main__":
    main()
