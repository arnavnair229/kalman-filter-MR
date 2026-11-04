"""Pairs backtest on pinned data. Signals use closes through t; fills happen at the close of t+1."""
from dataclasses import dataclass, asdict
from itertools import product
import numpy as np
import pandas as pd

TRADING_DAYS = 252
BORROW_ANNUAL = 0.005  # charged on the short leg


@dataclass(frozen=True)
class Params:
    q: float = 1e-4
    window: int = 60
    entry: float = 2.0
    exit: float = 0.5
    stop: float = 3.5


GRID = [Params(q, w, e, x, s) for q, w, e, x, s in product(
    (1e-5, 1e-4, 1e-3), (30, 60, 120), (1.5, 2.0, 2.5), (0.0, 0.5), (3.5,))]


def zscore(spread: pd.Series, window: int) -> pd.Series:
    """Compare today's spread with the mean and std of the previous `window` days."""
    past = spread.shift(1).rolling(window)
    return (spread - past.mean()) / past.std()


def target_positions(z: pd.Series, p: Params) -> pd.Series:
    """+1 is long y1 / short y2, -1 the reverse, 0 flat. Stay flat after a stop until z resets."""
    pos, out, locked = 0, [], False
    for v in z.to_numpy():
        if np.isnan(v):
            out.append(0)
            continue
        if locked and abs(v) < p.exit:
            locked = False
        if pos == 0 and not locked:
            if v > p.entry and v < p.stop:
                pos = -1
            elif v < -p.entry and v > -p.stop:
                pos = 1
        elif pos != 0:
            if abs(v) > p.stop:
                pos, locked = 0, True
            elif (pos == 1 and v > -p.exit) or (pos == -1 and v < p.exit):
                pos = 0
        out.append(pos)
    return pd.Series(out, index=z.index)


def simulate(px: pd.DataFrame, filt: pd.DataFrame, target: pd.Series, cost_bps: float) -> pd.DataFrame:
    """Dollar weights are w1 = pos/(1+|g|) and w2 = -pos*g/(1+|g|), so gross exposure is 1.
    The hedge ratio is frozen at the value known on the signal day and held until the position changes."""
    held = target.shift(1).fillna(0).astype(int)  # signal at close t, in the market from close t+1
    g = filt["gamma_prior"]
    w1, w2 = np.zeros(len(px)), np.zeros(len(px))
    for i in range(1, len(px)):
        pos = held.iloc[i]
        if pos == 0:
            continue
        if pos == held.iloc[i - 1]:
            w1[i], w2[i] = w1[i - 1], w2[i - 1]
        else:
            gamma = g.iloc[i - 1]
            denom = 1 + abs(gamma)
            w1[i], w2[i] = pos / denom, -pos * gamma / denom
    w = pd.DataFrame({"w1": w1, "w2": w2}, index=px.index)
    return pnl_from_weights(px, w, held, cost_bps)


def pnl_from_weights(px: pd.DataFrame, w: pd.DataFrame, held: pd.Series, cost_bps: float) -> pd.DataFrame:
    """Weights decided for day i earn from day i+1. Costs are charged on every change in weights."""
    rets = px.pct_change().fillna(0.0)
    active = w.shift(1).fillna(0.0)
    gross = active["w1"] * rets.iloc[:, 0] + active["w2"] * rets.iloc[:, 1]
    turnover = w.diff().abs().sum(axis=1).fillna(0.0)
    cost = turnover * cost_bps / 1e4
    short = active.clip(upper=0).abs().sum(axis=1)
    borrow = short * BORROW_ANNUAL / TRADING_DAYS
    return pd.DataFrame({"net": gross - cost - borrow, "gross": gross, "pos": held,
                         "w1": w["w1"], "w2": w["w2"]})


def trades(result: pd.DataFrame) -> list:
    """Round trips with net compounded return, including entry and exit costs."""
    out, start = [], None
    pos = result["pos"].to_numpy()
    net = result["net"].to_numpy()
    for i in range(1, len(pos)):
        if pos[i] != pos[i - 1]:
            if start is not None:
                out.append(float(np.prod(1 + net[start:i + 1]) - 1))
                start = None
            if pos[i] != 0:
                start = i
    if start is not None:
        out.append(float(np.prod(1 + net[start:]) - 1))
    return out


def metrics(result: pd.DataFrame) -> dict:
    r = result["net"]
    eq = (1 + r).cumprod()
    years = len(r) / TRADING_DAYS
    tr = trades(result)
    wins = [t for t in tr if t > 0]
    losses = [t for t in tr if t <= 0]
    sd = r.std()
    return {
        "days": int(len(r)),
        "total_return": float(eq.iloc[-1] - 1),
        "cagr": float(eq.iloc[-1] ** (1 / years) - 1) if eq.iloc[-1] > 0 else None,
        "ann_vol": float(sd * np.sqrt(TRADING_DAYS)),
        "sharpe": float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0,
        "max_drawdown": float((eq / eq.cummax() - 1).min()),
        "round_trips": len(tr),
        "win_rate": len(wins) / len(tr) if tr else None,
        "avg_win": float(np.mean(wins)) if wins else None,
        "avg_loss": float(np.mean(losses)) if losses else None,
        "pct_days_in_market": float((result["pos"] != 0).mean()),
    }


def sharpe_ci(r: pd.Series, n: int = 5000, block: int = 21, seed: int = 7) -> list:
    """Moving-block bootstrap 95% CI for the annualized Sharpe ratio."""
    rng = np.random.default_rng(seed)
    a = r.to_numpy()
    k = int(np.ceil(len(a) / block))
    stats = []
    for _ in range(n):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.integers(0, len(a) - block, k)])[: len(a)]
        s = a[idx]
        stats.append(s.mean() / s.std() * np.sqrt(TRADING_DAYS) if s.std() > 0 else 0.0)
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def to_dict(p: Params) -> dict:
    return asdict(p)
