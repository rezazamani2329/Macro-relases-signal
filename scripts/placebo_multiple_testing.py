"""Placebo and multiple-testing checks for the post-release drift (paper Table: tab_placebo).

Run from the repo root:  python scripts/placebo_multiple_testing.py
Needs the licensed inputs in data/ (not in the repository, see .gitignore) and the walk-forward
trade logs in results/. Output: results/placebo_drift.csv and results/multiple_testing.csv.

Drift D = 1e4 * log(P[T+30] / P[T+1]), the return a trade entering one minute after the release earns.

Placebo tests (CPI, NFP and PCE releases, 2016-2026; z = surprise signal standardized within family,
signed so that positive z predicts a positive first-minute return in that market):
  1. Permutation: the slope of D on z (family fixed effects) is compared with 10,000 slopes obtained
     after shuffling z across releases of the same family. p = share of shuffled |slopes| >= actual.
  2. Pseudo-release dates: each release's z is assigned to the same clock time one and two weeks
     before and after the release, on days with no scheduled release in our calendar, and D is
     measured there. If the drift is caused by the news, the slope at pseudo dates is zero.
Multiple testing:
  3. Equity pool: D on z for ES, NQ and YM stacked, with market x family fixed effects and standard
     errors clustered by release date (the same release is one observation, not three).
  4. Strategy tests: the alphas of Table tab_alpha (11 tests) with Holm and Benjamini-Hochberg
     adjusted p-values.
  5. Family tests: the slope of D on the signal x for every release family traded by the broad book
     (2019-2026, at least 15 single-family releases) in each market, with BH q-values.
"""
import glob
import numpy as np
import pandas as pd
from math import erf, sqrt

RNG = np.random.default_rng(20261112)
MARKETS = {"ES": "", "NQ": "_NQ", "YM": "_YM", "ZN": "_ZN"}
BARS = "data/processed/futures_1min_clean_v3_{m}.parquet"
CAL = "data/processed/macro_event_calendar_expanded_2010_2026.parquet"


def pval(t):
    return 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))


def ols(y, X, cluster=None):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    n, k = X.shape
    XtX = np.linalg.inv(X.T @ X)
    if cluster is None:  # HC1
        meat = (X.T * e**2) @ X
        V = XtX @ meat @ XtX * n / (n - k)
    else:
        g = pd.factorize(cluster)[0]
        G = g.max() + 1
        S = np.zeros((G, k))
        np.add.at(S, g, X * e[:, None])
        V = XtX @ (S.T @ S) @ XtX * (G / (G - 1)) * ((n - 1) / (n - k))
    return b, b / np.sqrt(np.diag(V))


def design(z, fam):
    F = pd.get_dummies(pd.Series(fam).astype(str), drop_first=True).astype(float).values
    return np.column_stack([np.ones(len(z)), z, F])


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for i, j in enumerate(o):
        run = max(run, (m - i) * p[j]); adj[j] = min(1, run)
    return adj


def bh(p):
    p = np.asarray(p); o = np.argsort(p)[::-1]; m = len(p); adj = np.empty(m); run = 1
    for i, j in enumerate(o):
        run = min(run, p[j] * m / (m - i)); adj[j] = run
    return adj


def load_events(m):
    d = pd.read_csv(f"data/processed/event_state_features_2016_2026{MARKETS[m]}.csv")
    d = d.dropna(subset=["x", "drift_30m_bps"]).copy()
    d["ts"] = pd.to_datetime(d.timestamp_utc, utc=True)
    d["z"] = d.x / d.groupby("event_type").x.transform("std")
    return d


def load_prices(m):
    p = pd.read_parquet(BARS.format(m=m))
    if "timestamp_utc" in p.columns:
        p = p.set_index("timestamp_utc")
    p.index = pd.to_datetime(p.index, utc=True)
    col = [c for c in p.columns if c.lower() == "close"][0]
    return p[col].astype(float).sort_index()


def drift_at(close, ts):
    """1e4*log(P[T+30]/P[T+1]) with P[t] = last close at or before t (bars stamped at their close)."""
    idx = close.index
    def px(t):
        i = idx.searchsorted(t, side="right") - 1
        ok = (i >= 0)
        out = np.full(len(t), np.nan)
        out[ok] = close.values[i[ok]]
        stale = np.full(len(t), True)
        stale[ok] = (t[ok] - idx[i[ok]]) > pd.Timedelta(minutes=5)
        out[stale] = np.nan
        return out
    p1 = px(ts + pd.Timedelta(minutes=1 + SHIFT))
    p30 = px(ts + pd.Timedelta(minutes=30 + SHIFT))
    return 1e4 * np.log(p30 / p1)


SHIFT = 0  # set in calibrate(): bar-stamp convention that reproduces drift_30m_bps


def calibrate(close, ev):
    global SHIFT
    best = None
    for s in (-1, 0, 1):
        SHIFT = s
        dd = drift_at(close, pd.DatetimeIndex(ev.ts))
        err = np.nanmedian(np.abs(dd - ev.drift_30m_bps.values))
        if best is None or err < best[1]:
            best = (s, err)
    SHIFT = best[0]
    return best


def main():
    cal = pd.read_parquet(CAL)
    tcol = [c for c in cal.columns if "utc" in c.lower()][0]
    release_days = set(pd.to_datetime(cal[tcol], utc=True).dt.tz_convert("America/New_York").dt.date)

    rows = []
    stack = []
    for m in MARKETS:
        ev_all = load_events(m)
        close = load_prices(m)
        shift, err = calibrate(close, ev_all)
        if m != "ZN":
            stack.append(ev_all.assign(mkt=m))
        for sample in ["CPI, NFP, PCE", "CPI"]:
            ev = ev_all if sample != "CPI" else ev_all[ev_all.event_type == "CPI"]
            ev = ev.reset_index(drop=True)
            fam = ev.event_type.values
            y, X = ev.drift_30m_bps.values, design(ev.z.values, fam)
            b, t = ols(y, X)
            # 1. permutation within family
            perm = np.empty(10000)
            zz = ev.z.values.copy()
            groups = [np.where(fam == f)[0] for f in np.unique(fam)]
            for r in range(10000):
                zp = zz.copy()
                for g in groups:
                    zp[g] = zz[RNG.permutation(g)]
                Xp = X.copy(); Xp[:, 1] = zp
                perm[r] = np.linalg.lstsq(Xp, y, rcond=None)[0][1]
            p_perm = (np.abs(perm) >= abs(b[1])).mean()
            # 2. pseudo-release dates
            ps = []
            for wk in (-2, -1, 1, 2):
                pts = ev.ts + pd.Timedelta(weeks=wk)
                day = pts.dt.tz_convert("America/New_York").dt.date
                keep = (~day.isin(release_days)).values
                dd = drift_at(close, pd.DatetimeIndex(pts[keep]))
                ps.append(pd.DataFrame({"D": dd, "z": ev.z.values[keep], "fam": fam[keep],
                                        "orig": ev.ts.values[keep]}))
            ps = pd.concat(ps).dropna()
            bp, tp = ols(ps.D.values, design(ps.z.values, ps.fam.values), cluster=ps.orig.values)
            rows.append(dict(market=m, sample=sample, n=len(ev), slope=b[1], t_hc1=t[1], p_perm=p_perm,
                             n_pseudo=len(ps), slope_pseudo=bp[1], t_pseudo=tp[1],
                             sd_drift=ev.drift_30m_bps.std(), sd_drift_pseudo=ps.D.std(),
                             calib_shift=shift, calib_median_abs_err=err))
    res = pd.DataFrame(rows)

    # 3. equity pool clustered by release date
    s = pd.concat(stack)
    X = design(s.z.values, (s.mkt + "_" + s.event_type).values)
    b, t_hc = ols(s.drift_30m_bps.values, X)
    _, t_cl = ols(s.drift_30m_bps.values, X, cluster=s.ts.values)
    res = pd.concat([res, pd.DataFrame([dict(market="ES+NQ+YM pooled", sample="CPI, NFP, PCE", n=len(s), slope=b[1], t_hc1=t_hc[1],
                                             t_cluster_date=t_cl[1])])])
    print(res.round(3).to_string(index=False))
    res.to_csv("results/placebo_drift.csv", index=False)

    # 4. strategy alphas, Holm and BH
    a = pd.read_csv("results/alpha_regression.csv")
    a["p"] = a.t_alpha.map(pval)
    a["p_holm"] = holm(a.p); a["q_bh"] = bh(a.p)
    a["test"] = "strategy alpha"
    # 5. family slopes in the broad book
    fam_rows = []
    for m, f in [("ES", "nb05_"), ("NQ", "nb05b_NQ_"), ("YM", "nb05b_YM_"), ("ZN", "nb05b_ZN_")]:
        d = pd.read_csv(f"results/{f}trade_log.csv")
        d = d[(d.strategy == "All · naive") & ~d.families.str.contains(r"\+", regex=True)]
        d = d[pd.to_datetime(d.timestamp_utc) >= "2019-01-01"].dropna(subset=["x", "ret_bps"])
        for fam, g in d.groupby("families"):
            if len(g) < 15 or g.x.std() == 0:
                continue
            bb, tt = ols(g.ret_bps.values, np.column_stack([np.ones(len(g)), g.x / g.x.std()]))
            fam_rows.append(dict(market=m, strategy=fam, trades=len(g), alpha_bps=bb[1], t_alpha=tt[1]))
    fr = pd.DataFrame(fam_rows)
    fr["p"] = fr.t_alpha.map(pval); fr["q_bh"] = bh(fr.p); fr["p_holm"] = holm(fr.p)
    fr["test"] = "family drift slope (bps per SD)"
    mt = pd.concat([a[["test", "market", "strategy", "trades", "alpha_bps", "t_alpha", "p", "p_holm", "q_bh"]],
                    fr[["test", "market", "strategy", "trades", "alpha_bps", "t_alpha", "p", "p_holm", "q_bh"]]])
    print(mt.round(3).to_string(index=False))
    mt.to_csv("results/multiple_testing.csv", index=False)


if __name__ == "__main__":
    main()
