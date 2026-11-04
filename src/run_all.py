"""Reproduce every number and figure in the README from data/prices.csv."""
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, coint

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kalman import run_filter
from backtest import GRID, Params, zscore, target_positions, simulate, pnl_from_weights, metrics, sharpe_ci, to_dict

warnings.simplefilter("ignore", FutureWarning)
ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FORMATION_DAYS, TEST_DAYS = 504, 63
COST_BPS = 5.0


def build(px, y1, y2, p, cost):
    filt = run_filter(y1, y2, q=p.q)
    z = zscore(filt["spread"], p.window)
    return filt, z, simulate(px, filt, target_positions(z, p), cost)


def walk_forward(px, y1, y2, cost):
    """Each quarter, pick the Sharpe-best params on the prior two years, then trade the next quarter."""
    runs = {p: build(px, y1, y2, p, cost)[2] for p in GRID}
    chunks, picks = [], []
    for start in range(FORMATION_DAYS, len(px), TEST_DAYS):
        form = (start - FORMATION_DAYS, start)
        best = max(GRID, key=lambda p: metrics(runs[p].iloc[form[0]:form[1]])["sharpe"])
        chunks.append(runs[best].iloc[start:start + TEST_DAYS])
        picks.append({"from": str(px.index[start].date()), **to_dict(best)})
    # Re-price the stitched weights so a parameter switch at a quarter seam pays its turnover.
    st = pd.concat(chunks)
    return pnl_from_weights(px.loc[st.index], st[["w1", "w2"]], st["pos"], cost), picks


def main():
    px = pd.read_csv(ROOT / "data" / "prices.csv", index_col="date", parse_dates=True)
    y1, y2 = np.log(px["EWH"]), np.log(px["EWZ"])

    split = px.index[FORMATION_DAYS]
    diag = {}
    for name, a, b in (("formation", y1[:split], y2[:split]), ("full_sample", y1, y2)):
        diag[name] = {
            "engle_granger_p": float(coint(a, b)[1]),
            "adf_ewh_level_p": float(adfuller(a)[1]),
            "adf_ewz_level_p": float(adfuller(b)[1]),
        }

    base = Params()
    filt, z, res = build(px, y1, y2, base, COST_BPS)
    wf, picks = walk_forward(px, y1, y2, COST_BPS)
    wf_free = walk_forward(px, y1, y2, 0.0)[0]
    wf_high = walk_forward(px, y1, y2, 10.0)[0]

    # Quantify the bug this version removes: same rules, spread built from the updated state.
    leaky = filt["mu_post"] + filt["gamma_post"] * y2
    leaky_spread = y1 - leaky
    leaky_res = simulate(px, filt, target_positions(zscore(leaky_spread, base.window), base), COST_BPS)

    out = {
        "data": {"pair": "EWH/EWZ", "start": str(px.index[0].date()), "end": str(px.index[-1].date())},
        "cointegration": diag,
        "assumptions": {"cost_bps_per_side": COST_BPS, "short_borrow_annual": 0.005,
                        "fill": "close of t+1", "formation_days": FORMATION_DAYS, "test_days": TEST_DAYS},
        "walk_forward_5bps": {**metrics(wf), "sharpe_95ci": sharpe_ci(wf["net"])},
        "walk_forward_0bps": metrics(wf_free),
        "walk_forward_10bps": metrics(wf_high),
        "fixed_params_full_sample_5bps": metrics(res),
        "lookahead_spread_full_sample_5bps": metrics(leaky_res),
        "param_picks": picks,
    }
    (RES / "metrics.json").write_text(json.dumps(out, indent=2))

    eq = (1 + wf["net"]).cumprod()
    export = {
        "repo": "https://github.com/arnavnair229/kalman-filter-MR",
        "pair": "EWH/EWZ",
        "metrics": {k: out["walk_forward_5bps"][k] for k in
                    ("cagr", "sharpe", "sharpe_95ci", "max_drawdown", "round_trips", "win_rate", "ann_vol")},
        "metrics_0bps": {k: out["walk_forward_0bps"][k] for k in ("cagr", "sharpe")},
        "metrics_10bps": {k: out["walk_forward_10bps"][k] for k in ("cagr", "sharpe")},
        "eg_p_formation": diag["formation"]["engle_granger_p"],
        "assumptions": out["assumptions"],
        "equity": [{"d": str(d.date()), "v": round(float(v), 4)} for d, v in eq.iloc[::3].items()],
        "gamma": [{"d": str(d.date()), "v": round(float(v), 4)} for d, v in filt["gamma_prior"].iloc[60::3].items()],
    }
    (RES / "site-export.json").write_text(json.dumps(export))

    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax[0].plot(filt.index[60:], filt["gamma_prior"].iloc[60:]); ax[0].set_ylabel("gamma (prior)")
    ax[1].plot(filt.index[60:], filt["mu_prior"].iloc[60:]); ax[1].set_ylabel("mu (prior)")
    fig.tight_layout(); fig.savefig(RES / "mu_gamma.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(z.index, z); ax.axhline(2, ls="--", c="grey"); ax.axhline(-2, ls="--", c="grey")
    ax.set_ylabel("spread z-score"); fig.tight_layout(); fig.savefig(RES / "Stationary.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(eq.index, eq); ax.axhline(1, c="grey", lw=.8)
    ax.set_ylabel("growth of 1, walk-forward, 5 bps"); fig.tight_layout()
    fig.savefig(RES / "walk_forward_equity.png", dpi=120); plt.close(fig)

    print(json.dumps({k: v for k, v in out.items() if k != "param_picks"}, indent=1))


if __name__ == "__main__":
    main()
