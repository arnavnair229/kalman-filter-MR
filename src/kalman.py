"""Dynamic-hedge Kalman filter for y1_t = mu_t + gamma_t * y2_t + e_t, with (mu, gamma) a random walk.

The spread is the one-step-ahead innovation: y1_t minus the prediction from the state known at t-1.
Using the updated state instead would let today's price into today's spread (look-ahead).
"""
import numpy as np
import pandas as pd


def causal_obs_noise(y1: pd.Series, window: int = 50, floor: float = 1e-6) -> np.ndarray:
    """Rolling variance of y1 returns using data up to t-1 only. The warm-up uses a fixed floor,
    never a backfill from future observations."""
    return y1.diff().rolling(window).var().shift(1).fillna(floor).clip(lower=floor).to_numpy()


def run_filter(y1: pd.Series, y2: pd.Series, q: float = 1e-4, r_window: int = 50) -> pd.DataFrame:
    R = causal_obs_noise(y1, r_window)
    x = np.array([0.0, 1.0])
    P = np.eye(2)
    Q = np.eye(2) * q
    rows = []
    for t, (a, b) in enumerate(zip(y1.to_numpy(), y2.to_numpy())):
        P_pred = P + Q
        H = np.array([1.0, b])
        # Prior: everything here was knowable before seeing today's y1.
        mu_prior, gamma_prior = x
        innovation = a - H @ x
        S = H @ P_pred @ H + R[t]
        K = P_pred @ H / S
        x = x + K * innovation
        P = P_pred - np.outer(K, H) @ P_pred
        rows.append((mu_prior, gamma_prior, innovation, S, x[0], x[1]))
    cols = ["mu_prior", "gamma_prior", "spread", "S", "mu_post", "gamma_post"]
    return pd.DataFrame(rows, index=y1.index, columns=cols)
