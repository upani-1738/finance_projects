"""
Symbolic-regression alpha discovery with VaR-constrained portfolio construction.
Single-file edition. Everything from the modular version, same code, one import.

    pip install numpy pandas scipy scikit-learn cvxpy matplotlib yfinance

    python qsr_system.py --synthetic-null     # must find ~nothing
    python qsr_system.py --synthetic-alpha    # must find the planted signal
    python qsr_system.py --real               # 50 S&P names + VIX via yfinance
    python qsr_system.py --self-test          # harness validation assertions

Run the two synthetic modes BEFORE the real one. A pipeline that reports a Sharpe
of 1.5 on pure noise is broken, and a real-data backtest cannot tell you that.

Contents
    1. Configuration
    2. Cross-sectional array math
    3. Data: yfinance download + synthetic panel generator
    4. Feature engineering (VaR/ES, vol structure, momentum, beta, BS hedge cost)
    5. Symbolic regression: tree-based genetic programming
    6. Selection, decorrelation, orthogonalization, Grinold alpha scaling
    7. Risk engine: shrinkage covariance, filtered historical simulation, VaR tests
    8. VaR-constrained conic portfolio optimizer, vol targeting, transaction costs
    9. Performance metrics incl. Probabilistic and Deflated Sharpe
   10. Purged + embargoed walk-forward backtest
   11. Reporting and artifacts
   12. CLI
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.stats import chi2, norm, skew, kurtosis
from sklearn.covariance import LedoitWolf

# ============================================================================
# CONFIGURATION
# ============================================================================
#
# Central configuration. Every number that matters lives here, not buried in code.
#

from dataclasses import dataclass, field
from typing import List

SP50 = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK-B", "JPM", "JNJ",
    "V", "PG", "UNH", "HD", "MA", "XOM", "CVX", "LLY", "ABBV", "PFE",
    "MRK", "KO", "PEP", "COST", "WMT", "BAC", "WFC", "GS", "MS", "CSCO",
    "ADBE", "CRM", "AMD", "INTC", "QCOM", "TXN", "ORCL", "IBM", "NFLX", "DIS",
    "NKE", "MCD", "SBUX", "CAT", "DE", "HON", "UNP", "LIN", "T", "VZ",
]


@dataclass
class Config:
    # ---------- universe / data ----------
    tickers: List[str] = field(default_factory=lambda: list(SP50))
    benchmark: str = "SPY"
    vix_ticker: str = "^VIX"
    start: str = "2018-01-01"
    end: str = "2024-12-31"
    cache_dir: str = "data"

    # ---------- labels ----------
    horizon: int = 5              # forward return horizon in trading days
    label_vol_normalize: bool = True   # divide fwd return by trailing vol before ranking

    # ---------- walk-forward ----------
    train_days: int = 504         # ~2y
    test_days: int = 21           # ~1m
    embargo_days: int = 5         # gap between train end and test start (>= horizon)
    inner_val_frac: float = 0.25  # tail of train window held out for model selection

    # ---------- symbolic regression (GP) ----------
    pop_size: int = 600
    generations: int = 20
    tournament_size: int = 7
    max_depth: int = 5
    max_nodes: int = 30
    p_crossover: float = 0.70
    p_subtree_mut: float = 0.12
    p_point_mut: float = 0.08
    p_hoist_mut: float = 0.05
    n_elite: int = 6
    const_range: tuple = (-2.0, 2.0)
    parsimony: float = 0.004      # penalty per node, in IR units
    turnover_lambda: float = 0.60 # penalty per unit one-way turnover, in IR units
    allow_oscillatory_ops: bool = False  # sin/cos/exp -- off by default, see README

    # ---------- ensemble ----------
    n_survivors: int = 40         # candidates carried from GP into selection
    max_pair_corr: float = 0.70   # decorrelation threshold for ensemble members
    min_val_ic: float = 0.005     # candidate must clear this on inner validation
    max_ensemble: int = 8

    # ---------- alpha scaling (Grinold) ----------
    ic_cap: float = 0.05          # cap the IC used to scale alphas; humility
    alpha_shrink: float = 0.5     # global shrink toward zero

    # ---------- portfolio ----------
    long_only: bool = False       # False => dollar-neutral long/short
    max_weight: float = 0.15
    gross_leverage: float = 1.0
    risk_aversion: float = 8.0
    turnover_cost_kappa: float = 0.0015   # L1 penalty in optimizer objective
    var_limit_daily: float = 0.020        # 1-day 95% VaR budget, fraction of capital
    var_limit_stress: float = 0.015       # tightened budget when VIX >= vix_stress
    vix_stress: float = 25.0
    target_ann_vol: float = 0.10          # post-optimization vol targeting
    max_gross_after_scaling: float = 2.0
    cov_lookback: int = 252
    rebalance_every: int = 5              # trading days; match the label horizon

    # ---------- costs ----------
    spread_bps: float = 5.0               # one-way, half-spread + fees
    impact_coef: float = 0.10             # sqrt-impact coefficient
    adv_participation: float = 0.02       # assumed participation for impact scaling

    # ---------- risk model ----------
    var_alpha: float = 0.05
    fhs_paths: int = 20000
    ewma_lambda: float = 0.94

    seed: int = 7


# ============================================================================
# CROSS-SECTIONAL ARRAY MATH
# ============================================================================
#
# Panel = dict of (T, N) float arrays sharing a date index and ticker axis.
#
# Everything downstream is cross-sectional, so a 2-D (time x name) layout with NaNs
# is far cleaner and ~100x faster than a long MultiIndex DataFrame.
#

import numpy as np


def rank_rows(x: np.ndarray) -> np.ndarray:
    """Per-row rank scaled to [-0.5, 0.5]. NaNs preserved. Rows with <2 obs -> NaN."""
    x = np.asarray(x, dtype=float)
    nan = ~np.isfinite(x)
    xf = np.where(nan, np.inf, x)
    order = np.argsort(xf, axis=1, kind="stable")
    ranks = np.empty(order.shape, dtype=float)
    rows = np.arange(x.shape[0])[:, None]
    ranks[rows, order] = np.arange(x.shape[1], dtype=float)[None, :]
    n = (~nan).sum(axis=1, keepdims=True).astype(float)
    ranks = np.where(nan, np.nan, ranks)
    out = ranks / np.maximum(n - 1.0, 1.0) - 0.5
    return np.where(n < 2, np.nan, out)


def zscore_rows(x: np.ndarray, clip: float = 4.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x, np.nan)
    import warnings
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(x, axis=1, keepdims=True)
        sd = np.nanstd(x, axis=1, keepdims=True)
    z = (x - mu) / np.where(sd > 1e-12, sd, np.nan)
    return np.clip(z, -clip, clip)


def row_corr(a: np.ndarray, b: np.ndarray, min_obs: int = 10) -> np.ndarray:
    """Pearson correlation per row, NaN-aware. Feed ranks in for Spearman."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    n = m.sum(axis=1)
    A = np.where(m, a, 0.0)
    B = np.where(m, b, 0.0)
    nn = np.maximum(n, 1)[:, None]
    A = np.where(m, A - A.sum(1, keepdims=True) / nn, 0.0)
    B = np.where(m, B - B.sum(1, keepdims=True) / nn, 0.0)
    num = (A * B).sum(axis=1)
    den = np.sqrt((A * A).sum(axis=1) * (B * B).sum(axis=1))
    ok = (den > 1e-12) & (n >= min_obs)
    return np.where(ok, num / np.where(den > 1e-12, den, 1.0), np.nan)


def signal_to_weights(sig: np.ndarray, gross: float = 1.0) -> np.ndarray:
    """Cross-sectionally demeaned, L1-normalized weights. Used only for turnover
    accounting inside the GP fitness -- the real optimizer runs downstream."""
    r = rank_rows(sig)
    r = np.where(np.isfinite(r), r, 0.0)
    r = r - r.mean(axis=1, keepdims=True)
    l1 = np.abs(r).sum(axis=1, keepdims=True)
    return np.divide(r, np.where(l1 > 1e-12, l1, 1.0)) * gross


def mean_turnover(sig: np.ndarray, every: int = 1) -> float:
    """One-way turnover per rebalance, in [0, 1]."""
    w = signal_to_weights(sig)[::every]
    if w.shape[0] < 2:
        return 0.0
    return float(np.abs(np.diff(w, axis=0)).sum(axis=1).mean() / 2.0)


def safe(x: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(x), x, np.nan)


# ============================================================================
# DATA: DOWNLOAD + SYNTHETIC PANEL
# ============================================================================
#
# Data acquisition. Real data via yfinance; synthetic data for harness validation.
#
# The synthetic generator is not a toy -- it is how you prove the pipeline is not
# lying to you. Run the whole system on a panel with ZERO alpha. If it reports a
# Sharpe of 1.5, your backtest is broken, not your market.
#

import os
import numpy as np
import pandas as pd



# ----------------------------------------------------------------------------- real data
def download_data(cfg: Config, force: bool = False) -> dict:
    """Returns dict with 'close', 'volume' (DataFrames T x N), 'vix', 'bench' (Series)."""
    os.makedirs(cfg.cache_dir, exist_ok=True)
    px_p = os.path.join(cfg.cache_dir, "close.csv")
    vo_p = os.path.join(cfg.cache_dir, "volume.csv")
    mk_p = os.path.join(cfg.cache_dir, "market.csv")

    if not force and all(os.path.exists(p) for p in (px_p, vo_p, mk_p)):
        close = pd.read_csv(px_p, index_col=0, parse_dates=True)
        volume = pd.read_csv(vo_p, index_col=0, parse_dates=True)
        mkt = pd.read_csv(mk_p, index_col=0, parse_dates=True)
    else:
        import yfinance as yf  # imported lazily so synthetic mode needs no network

        syms = list(cfg.tickers) + [cfg.benchmark, cfg.vix_ticker]
        raw = yf.download(
            syms, start=cfg.start, end=cfg.end,
            auto_adjust=True, progress=False, group_by="column", threads=True,
        )
        close = raw["Close"].copy()
        volume = raw["Volume"].copy()
        mkt = pd.DataFrame({
            "bench": close[cfg.benchmark],
            "vix": close[cfg.vix_ticker],
        })
        close = close[[t for t in cfg.tickers if t in close.columns]]
        volume = volume[[c for c in close.columns if c in volume.columns]]
        close.to_csv(px_p); volume.to_csv(vo_p); mkt.to_csv(mk_p)

    # Drop names with too little history rather than forward-filling a fiction.
    good = close.columns[close.notna().mean() > 0.95]
    close, volume = close[good], volume[good]
    close = close.ffill(limit=3)
    return {"close": close, "volume": volume,
            "vix": mkt["vix"].ffill(), "bench": mkt["bench"].ffill()}


# ----------------------------------------------------------------------------- synthetic
def synthetic_panel(n_days: int = 1600, n_names: int = 50, seed: int = 0,
              planted_ic: float = 0.0) -> dict:
    """Fat-tailed, vol-clustered, factor-driven panel.

    planted_ic > 0 injects a genuine predictive relationship between a *known*
    function of the features and forward returns, so you can check the pipeline
    can actually find something when something is there.
    planted_ic == 0 is the null: correct behaviour is to find nothing.

    The market factor is generated with ZERO drift on purpose. With positive drift
    and dispersed betas, a signal that simply tilts toward high beta earns positive
    cross-sectional IC -- a real exposure, not a leak, but it means the panel is not
    actually null and the harness check would be measuring the wrong thing. (This is
    also, in miniature, the reason ensemble orthogonalization exists.)
    """
    rng = np.random.default_rng(seed)
    T, N = n_days, n_names

    # market factor with GARCH-ish vol clustering and t-distributed shocks
    vol = np.zeros(T); vol[0] = 0.011
    mshock = rng.standard_t(df=4, size=T) / np.sqrt(2.0)
    mkt = np.zeros(T)
    for t in range(1, T):
        vol[t] = np.sqrt(2e-6 + 0.10 * mkt[t - 1] ** 2 + 0.87 * vol[t - 1] ** 2)
        mkt[t] = vol[t] * mshock[t]   # zero drift -- see note below

    beta = rng.uniform(0.5, 1.6, size=N)
    ivol = rng.uniform(0.008, 0.030, size=N)
    idio = rng.standard_t(df=5, size=(T, N)) / np.sqrt(5 / 3) * ivol
    rets = mkt[:, None] * beta[None, :] + idio

    if planted_ic > 0:
        # A *persistent* signal: names with negative 60-day idiosyncratic skew earn
        # a premium. Persistence matters -- a one-day-ahead signal is invisible to a
        # 5-day label. It is deliberately not one of the factors we orthogonalize
        # against, so a positive result means the search found something real rather
        # than rediscovering momentum.
        sk = pd.DataFrame(idio).rolling(60).skew().shift(1).to_numpy()
        z = (sk - np.nanmean(sk, axis=1, keepdims=True)) / (
            np.nanstd(sk, axis=1, keepdims=True) + 1e-12)
        rets = rets + planted_ic * np.nan_to_num(-z) * ivol[None, :]

    price = 50.0 * np.exp(np.cumsum(rets, axis=0))
    idx = pd.bdate_range("2018-01-02", periods=T)
    cols = [f"SYN{i:02d}" for i in range(N)]
    close = pd.DataFrame(price, index=idx, columns=cols)
    volume = pd.DataFrame(rng.lognormal(15, 0.6, size=(T, N)), index=idx, columns=cols)
    bench = pd.Series(100.0 * np.exp(np.cumsum(mkt)), index=idx)
    vix = pd.Series(np.clip(vol * np.sqrt(252) * 100 * 1.15, 9, 80), index=idx)
    return {"close": close, "volume": volume, "vix": vix, "bench": bench}


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
#
# Feature construction.
#
# Design rules that matter more than the specific features:
#
# 1. Every feature is cross-sectionally normalized before the GP sees it. A formula
#    fitted on raw VaR levels cannot transfer between a $20 stock and a $600 stock.
# 2. Every feature is strictly backward-looking and shifted so that the value on
#    day t uses information available at the close of day t.
# 3. VIX enters as a *time-series* z-score broadcast across names, not a
#    cross-sectional feature (it is identical for every stock, so cross-sectional
#    normalization would zero it out). This lets the GP discover regime
#    conditioning instead of us hard-coding a threshold at VIX = 25.
#
# On the two components from the original spec that are NOT here as features:
#
# * Monte Carlo GBM skewness / percentiles. Under GBM these are closed-form
#   functions of drift and sigma -- simulating 5,000 paths per name per day buys
#   you a slow, noisy estimate of something you can write down, and the resulting
#   columns are ~0.99 correlated with realized vol. Monte Carlo is retained in
#   risk.py where it earns its keep: filtered historical simulation for portfolio
#   tail risk.
# * Black-Scholes delta computed from *realized* vol. It does not describe "what
#   the options market is pricing in" -- there is no options data in it. For a
#   near-ATM option it is a squashed monotone function of vol, so it adds no
#   information beyond rvol. What is kept below is the one genuinely useful BS
#   output: the cost of insuring the position, which is a real economic quantity.
#

import numpy as np
import pandas as pd
from scipy.stats import norm



# ------------------------------------------------------------------ Black-Scholes
def bs_put(S, K, T, r, sigma, q=0.0):
    S, K, T, sigma = map(np.asarray, (S, K, T, sigma))
    sigma = np.maximum(sigma, 1e-6)
    T = np.maximum(T, 1e-6)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def bs_delta_call(S, K, T, r, sigma, q=0.0):
    sigma = np.maximum(np.asarray(sigma), 1e-6)
    T = np.maximum(np.asarray(T), 1e-6)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return np.exp(-q * T) * norm.cdf(d1)


# ------------------------------------------------------------------ feature panel
def build_features(raw: dict, cfg: Config, rf: float = 0.04) -> dict:
    close: pd.DataFrame = raw["close"]
    volume: pd.DataFrame = raw["volume"]
    bench: pd.Series = raw["bench"].reindex(close.index).ffill()
    vix: pd.Series = raw["vix"].reindex(close.index).ffill()

    ret = np.log(close).diff()
    bret = np.log(bench).diff()

    f: dict[str, pd.DataFrame] = {}

    # --- tail risk (historical, non-parametric) ---
    f["var95_60"] = ret.rolling(60).quantile(0.05)
    f["es95_60"] = ret.rolling(60).apply(
        lambda a: a[a <= np.quantile(a, 0.05)].mean(), raw=True)

    # --- volatility structure ---
    rv5 = ret.rolling(5).std()
    rv20 = ret.rolling(20).std()
    rv60 = ret.rolling(60).std()
    f["rvol_20"] = rv20
    f["vol_ratio"] = rv5 / (rv60 + 1e-8)          # compression vs expansion
    f["vol_of_vol"] = rv20.rolling(60).std() / (rv20 + 1e-8)
    f["skew_60"] = ret.rolling(60).skew()
    f["kurt_60"] = ret.rolling(60).kurt()

    # --- price / trend ---
    f["mom_21"] = np.log(close).diff(21)
    f["mom_252_21"] = np.log(close).diff(252).shift(21)   # 12-1 momentum
    f["rev_5"] = -np.log(close).diff(5)                   # short-term reversal
    f["dist_hi_252"] = close / close.rolling(252).max() - 1.0

    # --- market exposure ---
    cov = ret.rolling(60).cov(bret)
    f["beta_60"] = cov.div(bret.rolling(60).var(), axis=0)
    resid_var = (rv60 ** 2).sub(
        (f["beta_60"] ** 2).mul(bret.rolling(60).var(), axis=0), axis=0)
    f["ivol_60"] = np.sqrt(resid_var.clip(lower=0))

    # --- liquidity ---
    dv = (close * volume).rolling(20).mean()
    f["log_dollar_vol"] = np.log(dv.clip(lower=1.0))
    f["amihud"] = (ret.abs() / (close * volume + 1.0)).rolling(20).mean() * 1e9

    # --- Black-Scholes: cost of insuring one month, 5% out of the money ---
    ann = rv20 * np.sqrt(252)
    f["put_cost_5pct"] = pd.DataFrame(
        bs_put(close.to_numpy(), close.to_numpy() * 0.95, 21 / 252.0, rf, ann.to_numpy()),
        index=close.index, columns=close.columns) / close
    f["bs_delta_atm"] = pd.DataFrame(
        bs_delta_call(close.to_numpy(), close.to_numpy(), 21 / 252.0, rf, ann.to_numpy()),
        index=close.index, columns=close.columns)

    # Cross-sectionally normalize everything above, then shift by 1 day.
    feats = {k: zscore_rows(v.to_numpy()) for k, v in f.items()}
    feats = {k: np.vstack([np.full((1, v.shape[1]), np.nan), v[:-1]])
             for k, v in feats.items()}

    # VIX as a broadcast time-series z-score (expanding, so no lookahead).
    lv = np.log(vix.clip(lower=1.0))
    vz = ((lv - lv.expanding(250).mean()) / (lv.expanding(250).std() + 1e-8)).shift(1)
    feats["vix_z"] = np.repeat(vz.to_numpy()[:, None], close.shape[1], axis=1)

    return {
        "features": feats,
        "returns": ret.to_numpy(),
        "close": close.to_numpy(),
        "dollar_vol": dv.to_numpy(),
        "dates": close.index,
        "tickers": list(close.columns),
        "vix": vix.to_numpy(),
        "bench_ret": bret.to_numpy(),
    }


def make_labels(returns: np.ndarray, cfg: Config) -> np.ndarray:
    """Forward `horizon`-day return, vol-normalized, then cross-sectionally ranked.

    Ranking is not cosmetic. Raw forward returns are dominated by a handful of
    earnings-day outliers, and an MSE-style fit will spend its whole budget on
    them. Ranks also match how the signal is actually traded.
    """
    r = pd.DataFrame(returns)
    fwd = r.rolling(cfg.horizon).sum().shift(-cfg.horizon).to_numpy()
    if cfg.label_vol_normalize:
        vol = r.rolling(60).std().to_numpy() * np.sqrt(cfg.horizon)
        fwd = fwd / np.where(vol > 1e-8, vol, np.nan)
    return rank_rows(fwd)


# ============================================================================
# SYMBOLIC REGRESSION (GENETIC PROGRAMMING)
# ============================================================================
#
# Tree-based genetic programming for cross-sectional alpha discovery.
#
# Two departures from the original spec, both load-bearing:
#
# ONE FORMULA FOR THE WHOLE CROSS-SECTION, not one per stock. Fitting a separate
# expression to each of 50 names on 504 observations each is 50 independent
# searches over a huge hypothesis space with almost no data per search -- it is an
# overfitting machine, and it multiplies the multiple-testing problem by 50. A
# single cross-sectional formula sees 50x the observations, is regularized by
# having to work on every name at once, and produces exactly the long/short ranking
# you trade.
#
# FITNESS IS COST-ADJUSTED INFORMATION RATIO, not MSE and not Calmar. Calmar on a
# 504-day training window depends on a single max-drawdown observation; it is one
# of the noisiest statistics available and makes the fitness landscape jump around
# between generations. IC_mean / IC_std is estimated from ~500 daily observations
# and is stable. Turnover is charged inside fitness, because a signal the search
# cannot see the cost of is a signal the search will happily make unaffordable.
#

import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple



# ----------------------------------------------------------------- operator set
def _pdiv(a, b):
    return np.where(np.abs(b) > 1e-4, a / np.where(np.abs(b) > 1e-4, b, 1.0), 1.0)


def _plog(a):
    return np.log(np.abs(a) + 1e-6)


BASE_OPS: Dict[str, Tuple[int, Callable]] = {
    "add":  (2, np.add),
    "sub":  (2, np.subtract),
    "mul":  (2, np.multiply),
    "div":  (2, _pdiv),
    "min":  (2, np.minimum),
    "max":  (2, np.maximum),
    "neg":  (1, np.negative),
    "abs":  (1, np.abs),
    "sqrt": (1, lambda a: np.sqrt(np.abs(a))),
    "log":  (1, _plog),
    "sign": (1, np.sign),
    "rank": (1, rank_rows),     # cross-sectional
    "zsc":  (1, zscore_rows),   # cross-sectional
}

# Deliberately excluded by default. exp/pow blow up and fit outliers; sin/tan
# manufacture periodicity that does not exist and are unfalsifiable when they
# appear in a "discovered" formula. Enable only if you want to see this happen.
OSC_OPS: Dict[str, Tuple[int, Callable]] = {
    "sin": (1, np.sin),
    "cos": (1, np.cos),
    "exp": (1, lambda a: np.exp(np.clip(a, -8, 8))),
}


# ----------------------------------------------------------------- expression tree
@dataclass
class Node:
    kind: str                    # 'op' | 'feat' | 'const'
    value: object                # op name | feature name | float
    children: List["Node"] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []


def _nodes(n: Node) -> List[Node]:
    out = [n]
    for c in n.children:
        out.extend(_nodes(c))
    return out


def size(n: Node) -> int:
    return len(_nodes(n))


def depth(n: Node) -> int:
    return 1 if not n.children else 1 + max(depth(c) for c in n.children)


def clone(n: Node) -> Node:
    return Node(n.kind, n.value, [clone(c) for c in n.children])


def to_str(n: Node) -> str:
    if n.kind == "feat":
        return str(n.value)
    if n.kind == "const":
        return f"{n.value:.3f}"
    a = [to_str(c) for c in n.children]
    infix = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
    if n.value in infix:
        return f"({a[0]} {infix[n.value]} {a[1]})"
    return f"{n.value}({', '.join(a)})"


def evaluate(n: Node, feats: Dict[str, np.ndarray], shape) -> np.ndarray:
    if n.kind == "feat":
        return feats[n.value]
    if n.kind == "const":
        return np.full(shape, float(n.value))
    arity, fn = n.op_impl
    args = [evaluate(c, feats, shape) for c in n.children]
    with np.errstate(all="ignore"):
        out = fn(*args)
    return np.where(np.isfinite(out), out, np.nan)


# ----------------------------------------------------------------- the engine
class SymbolicRegressor:
    def __init__(self, cfg: Config, feature_names: List[str], rng=None):
        self.cfg = cfg
        self.fnames = list(feature_names)
        self.rng = rng or np.random.default_rng(cfg.seed)
        self.ops = dict(BASE_OPS)
        if cfg.allow_oscillatory_ops:
            self.ops.update(OSC_OPS)
        self.op_names = list(self.ops)
        self.n_evaluated = 0          # trial count -> feeds the Deflated Sharpe Ratio
        self.seen: set = set()

    # ---------------- tree construction ----------------
    def _bind(self, n: Node) -> Node:
        if n.kind == "op":
            n.op_impl = self.ops[n.value]
        for c in n.children:
            self._bind(c)
        return n

    def _terminal(self) -> Node:
        if self.rng.random() < 0.85:
            return Node("feat", self.fnames[self.rng.integers(len(self.fnames))])
        lo, hi = self.cfg.const_range
        return Node("const", float(self.rng.uniform(lo, hi)))

    def _grow(self, d: int, full: bool) -> Node:
        if d <= 1 or (not full and self.rng.random() < 0.35):
            return self._terminal()
        name = self.op_names[self.rng.integers(len(self.op_names))]
        arity = self.ops[name][0]
        return Node("op", name, [self._grow(d - 1, full) for _ in range(arity)])

    def _random_tree(self) -> Node:
        d = int(self.rng.integers(2, self.cfg.max_depth + 1))
        return self._bind(self._grow(d, full=self.rng.random() < 0.5))

    # ---------------- genetic operators ----------------
    def _crossover(self, a: Node, b: Node) -> Node:
        a, b = clone(a), clone(b)
        na, nb = _nodes(a), _nodes(b)
        tgt = na[self.rng.integers(len(na))]
        src = clone(nb[self.rng.integers(len(nb))])
        tgt.kind, tgt.value, tgt.children = src.kind, src.value, src.children
        return self._bind(a)

    def _subtree_mut(self, a: Node) -> Node:
        a = clone(a)
        na = _nodes(a)
        tgt = na[self.rng.integers(len(na))]
        src = self._grow(int(self.rng.integers(1, 4)), full=False)
        tgt.kind, tgt.value, tgt.children = src.kind, src.value, src.children
        return self._bind(a)

    def _point_mut(self, a: Node) -> Node:
        a = clone(a)
        for nd in _nodes(a):
            if self.rng.random() > 0.25:
                continue
            if nd.kind == "feat":
                nd.value = self.fnames[self.rng.integers(len(self.fnames))]
            elif nd.kind == "const":
                nd.value = float(nd.value + self.rng.normal(0, 0.3))
            else:
                same = [o for o in self.op_names
                        if self.ops[o][0] == self.ops[nd.value][0]]
                nd.value = same[self.rng.integers(len(same))]
        return self._bind(a)

    def _hoist_mut(self, a: Node) -> Node:
        na = _nodes(clone(a))
        return self._bind(clone(na[self.rng.integers(len(na))]))

    def _legal(self, n: Node) -> bool:
        return size(n) <= self.cfg.max_nodes and depth(n) <= self.cfg.max_depth

    # ---------------- fitness ----------------
    def fitness(self, tree: Node, feats, label, shape, every: int) -> Tuple[float, float]:
        """Returns (penalized score, raw mean IC)."""
        try:
            sig = evaluate(tree, feats, shape)
        except Exception:
            return -1e9, np.nan
        if not np.isfinite(sig).any():
            return -1e9, np.nan
        # A signal with no cross-sectional dispersion is a constant: useless.
        disp = np.nanstd(sig, axis=1)
        if np.nanmean(disp) < 1e-10:
            return -1e9, np.nan

        ic = row_corr(rank_rows(sig), label)
        ic = ic[np.isfinite(ic)]
        if ic.size < 60:
            return -1e9, np.nan

        mean_ic, sd_ic = float(ic.mean()), float(ic.std())
        ir = mean_ic / (sd_ic + 1e-9)
        turn = mean_turnover(sig, every=every)
        score = (ir
                 - self.cfg.turnover_lambda * turn
                 - self.cfg.parsimony * size(tree))
        return float(score), mean_ic

    # ---------------- evolution ----------------
    def fit(self, feats: Dict[str, np.ndarray], label: np.ndarray,
            verbose: bool = False) -> List[Tuple[float, float, Node]]:
        cfg = self.cfg
        shape = label.shape
        every = cfg.rebalance_every

        pop = [self._random_tree() for _ in range(cfg.pop_size)]
        hall: Dict[str, Tuple[float, float, Node]] = {}

        for gen in range(cfg.generations):
            scored = []
            for t in pop:
                key = to_str(t)
                if key in self.seen and key in hall:
                    scored.append((hall[key][0], hall[key][1], t))
                    continue
                self.seen.add(key)
                self.n_evaluated += 1
                s, ic = self.fitness(t, feats, label, shape, every)
                scored.append((s, ic, t))
                if np.isfinite(s) and s > -1e8:
                    hall[key] = (s, ic, clone(t))
            scored.sort(key=lambda x: -x[0])

            if verbose:
                print(f"  gen {gen:02d}  best={scored[0][0]:+.3f} "
                      f"ic={scored[0][1]:+.4f}  {to_str(scored[0][2])[:70]}")

            if gen == cfg.generations - 1:
                break

            new = [clone(s[2]) for s in scored[:cfg.n_elite]]
            while len(new) < cfg.pop_size:
                r = self.rng.random()
                p1 = self._tournament(scored)
                if r < cfg.p_crossover:
                    child = self._crossover(p1, self._tournament(scored))
                elif r < cfg.p_crossover + cfg.p_subtree_mut:
                    child = self._subtree_mut(p1)
                elif r < cfg.p_crossover + cfg.p_subtree_mut + cfg.p_point_mut:
                    child = self._point_mut(p1)
                elif r < (cfg.p_crossover + cfg.p_subtree_mut
                          + cfg.p_point_mut + cfg.p_hoist_mut):
                    child = self._hoist_mut(p1)
                else:
                    child = clone(p1)
                new.append(child if self._legal(child) else self._random_tree())
            pop = [self._bind(t) for t in new]

        ranked = sorted(hall.values(), key=lambda x: -x[0])
        return ranked[:self.cfg.n_survivors]

    def _tournament(self, scored) -> Node:
        idx = self.rng.integers(0, len(scored), size=self.cfg.tournament_size)
        best = max((scored[i] for i in idx), key=lambda x: x[0])
        return best[2]

    def rebind(self, tree: Node) -> Node:
        return self._bind(tree)


# ============================================================================
# SELECTION, ENSEMBLING, ALPHA SCALING
# ============================================================================
#
# Turn a GP population into a tradable alpha.
#
# The single most damaging thing you can do after a symbolic regression run is take
# the top-1 expression. The top-1 is, almost by construction, the luckiest point in
# a search over ~10^5 candidates. What survives out of sample is the part of the
# population that agrees with itself.
#
# Pipeline: inner-validation screen -> decorrelate -> equal-weight ensemble ->
# orthogonalize against generic factors -> scale to expected returns via Grinold.
#

import numpy as np
from typing import Dict, List, Tuple



def _sig(tree, reg, feats, shape) -> np.ndarray:
    return evaluate(reg.rebind(tree), feats, shape)


def select_candidates(survivors, reg, feats: Dict[str, np.ndarray], label: np.ndarray,
           val_slice: slice, cfg: Config) -> Tuple[List[Node], List[float]]:
    """Screen candidates on a held-out tail of the training window, then decorrelate."""
    shape = label.shape
    scored = []
    for score, ic_tr, tree in survivors:
        s = _sig(tree, reg, feats, shape)
        ic = row_corr(rank_rows(s[val_slice]), label[val_slice])
        ic = ic[np.isfinite(ic)]
        if ic.size < 10:
            continue
        val_ic = float(ic.mean())
        if val_ic < cfg.min_val_ic:
            continue
        scored.append((val_ic, tree, rank_rows(s)))

    scored.sort(key=lambda x: -x[0])

    kept: List[Node] = []
    kept_ic: List[float] = []
    kept_sig: List[np.ndarray] = []
    for val_ic, tree, rs in scored:
        if len(kept) >= cfg.max_ensemble:
            break
        if any(_pooled_corr(rs, k) > cfg.max_pair_corr for k in kept_sig):
            continue
        kept.append(tree)
        kept_ic.append(val_ic)
        kept_sig.append(rs)
    return kept, kept_ic


def _pooled_corr(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 100:
        return 0.0
    x, y = a[m], b[m]
    x = x - x.mean(); y = y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float(abs((x * y).sum() / d)) if d > 0 else 0.0


def combine_signals(trees: List[Node], reg, feats: Dict[str, np.ndarray],
            shape) -> np.ndarray:
    """Equal-weight blend of rank-standardized member signals."""
    if not trees:
        return np.full(shape, np.nan)
    acc = np.zeros(shape)
    cnt = np.zeros(shape)
    for t in trees:
        r = rank_rows(_sig(t, reg, feats, shape))
        ok = np.isfinite(r)
        acc[ok] += r[ok]
        cnt[ok] += 1
    out = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    return zscore_rows(out)


def orthogonalize(sig: np.ndarray, feats: Dict[str, np.ndarray],
                  against=("beta_60", "mom_252_21", "rvol_20", "log_dollar_vol")) -> np.ndarray:
    """Row-wise regression of the signal on generic factors; keep the residual.

    Without this you can pay yourself a performance fee for momentum and low-vol
    exposure that anyone can buy in an ETF for a few basis points.
    """
    cols = [feats[k] for k in against if k in feats]
    if not cols:
        return sig
    T, N = sig.shape
    out = np.full_like(sig, np.nan)
    X_all = np.stack(cols, axis=2)                      # T x N x K
    for t in range(T):
        y = sig[t]
        X = X_all[t]
        m = np.isfinite(y) & np.isfinite(X).all(axis=1)
        if m.sum() < X.shape[1] + 5:
            continue
        A = np.column_stack([np.ones(m.sum()), X[m]])
        try:
            beta, *_ = np.linalg.lstsq(A, y[m], rcond=None)
        except np.linalg.LinAlgError:
            continue
        out[t, m] = y[m] - A @ beta
    return zscore_rows(out)


def to_expected_returns(sig: np.ndarray, fwd_vol: np.ndarray,
                        ic_hat: float, cfg: Config) -> np.ndarray:
    """Grinold: alpha = IC * volatility * score.

    This replaces the "0.6 * SR_prediction + 0.4 * historical_mean" blend from the
    original spec. Blending toward the historical mean return is not shrinkage
    toward something safe -- sample mean returns are among the noisiest estimates
    in finance, and mixing them in adds variance rather than removing it. Shrinking
    toward ZERO, and capping the IC you are willing to claim, is the honest version.
    """
    ic = float(np.clip(ic_hat, 0.0, cfg.ic_cap))
    z = np.clip(zscore_rows(sig), -3, 3)
    return cfg.alpha_shrink * ic * fwd_vol * z


def describe_formulas(trees: List[Node], ics: List[float]) -> str:
    lines = []
    for t, ic in zip(trees, ics):
        lines.append(f"    val_IC={ic:+.4f}  {to_str(t)}")
    return "\n".join(lines) if lines else "    (no candidate survived validation)"


# ============================================================================
# RISK ENGINE: COVARIANCE, MONTE CARLO, VaR TESTS
# ============================================================================
#
# Risk engine.
#
# This is where Monte Carlo belongs. Simulating GBM paths per stock to make
# features produces a slow restatement of realized vol. Simulating the *portfolio*
# under a filtered historical bootstrap gives you something GBM cannot: fat tails
# and volatility clustering, which are the only two properties of returns that
# matter for a tail risk number.
#
# Size of the effect: at the 95% level the gap is usually modest (single-digit to
# low-teens percent). It widens sharply as you go further out -- at the 99% level,
# and especially in clustered-volatility episodes, GBM commonly understates the
# loss by a third or more. Which is precisely when you needed the number.
#

import numpy as np
from scipy.stats import chi2, norm
from sklearn.covariance import LedoitWolf



# --------------------------------------------------------------------- covariance
def shrunk_cov(rets: np.ndarray, ewma_lambda: float | None = None) -> np.ndarray:
    """Ledoit-Wolf shrinkage, optionally on EWMA-weighted observations.

    With 50 names and a 60-day window the sample covariance is singular by
    construction -- the original spec's EWMA(60) matrix cannot be inverted, and
    an optimizer handed a near-singular matrix will produce enormous offsetting
    positions in correlated pairs. Shrinkage is not a refinement here, it is a
    correctness requirement.
    """
    X = rets[np.isfinite(rets).all(axis=1)]
    if X.shape[0] < 20:
        return np.eye(rets.shape[1]) * 1e-4
    if ewma_lambda is not None:
        T = X.shape[0]
        w = ewma_lambda ** np.arange(T - 1, -1, -1)
        w = w / w.sum()
        X = (X - (w[:, None] * X).sum(0)) * np.sqrt(w[:, None] * T)
    S = LedoitWolf(assume_centered=(ewma_lambda is not None)).fit(X).covariance_
    # numerical PSD repair
    ev, V = np.linalg.eigh((S + S.T) / 2)
    return V @ np.diag(np.clip(ev, 1e-12, None)) @ V.T


# --------------------------------------------------------------------- simulation
def ewma_vol(x: np.ndarray, lam: float = 0.94) -> np.ndarray:
    v = np.empty_like(x)
    var = np.var(x[:20]) if x.size >= 20 else np.var(x) + 1e-8
    for i, r in enumerate(x):
        var = lam * var + (1 - lam) * r ** 2
        v[i] = np.sqrt(max(var, 1e-12))
    return v


def filtered_historical_sim(port_rets: np.ndarray, horizon: int = 1,
                            n_paths: int = 20000, lam: float = 0.94,
                            rng=None) -> np.ndarray:
    """Bootstrap standardized residuals, re-inflate through the current vol level."""
    rng = rng or np.random.default_rng(0)
    x = port_rets[np.isfinite(port_rets)]
    if x.size < 60:
        return np.zeros(n_paths)
    vol = ewma_vol(x, lam)
    z = x / vol
    z = z[np.isfinite(z)]
    cur = vol[-1]
    draws = rng.choice(z, size=(n_paths, horizon), replace=True)
    return (draws * cur).sum(axis=1)


def gbm_sim(mu: float, sigma: float, horizon: int = 1,
            n_paths: int = 20000, rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    return (mu - 0.5 * sigma ** 2) * horizon + sigma * np.sqrt(horizon) * rng.standard_normal(n_paths)


def var_es(sample: np.ndarray, alpha: float = 0.05):
    q = float(np.quantile(sample, alpha))
    tail = sample[sample <= q]
    return -q, -float(tail.mean()) if tail.size else -q


# --------------------------------------------------------------------- BS overlay
def put_hedge_drag(spot_vol_ann: float, moneyness: float = 0.95,
                   tenor_days: int = 21, rf: float = 0.04,
                   vrp: float = 1.15) -> float:
    """Annualized cost of rolling a protective put, as a fraction of NAV.

    `vrp` marks realized vol up to a plausible implied vol -- index puts trade at
    a persistent premium, and pricing your hedge at realized vol will make tail
    protection look roughly a third cheaper than it is.
    """
    p = float(bs_put(1.0, moneyness, tenor_days / 252.0, rf, spot_vol_ann * vrp))
    return p * (252.0 / tenor_days)


# --------------------------------------------------------------------- VaR backtests
def kupiec_pof(exceedances: np.ndarray, alpha: float = 0.05):
    """Unconditional coverage test. H0: exceedance rate == alpha."""
    x = np.asarray(exceedances, dtype=bool)
    n, k = x.size, int(x.sum())
    if n == 0 or k == 0:
        return np.nan, np.nan
    pi = k / n
    ll0 = (n - k) * np.log(1 - alpha) + k * np.log(alpha)
    ll1 = (n - k) * np.log(1 - pi) + k * np.log(pi)
    lr = -2 * (ll0 - ll1)
    return float(lr), float(1 - chi2.cdf(lr, 1))


def christoffersen_independence(exceedances: np.ndarray):
    """H0: exceedances are not clustered. Clustering is the failure mode that
    actually bankrupts people -- five breaches in one week, not five per year."""
    x = np.asarray(exceedances, dtype=int)
    if x.size < 10:
        return np.nan, np.nan
    n00 = n01 = n10 = n11 = 0
    for a, b in zip(x[:-1], x[1:]):
        if a == 0 and b == 0: n00 += 1
        elif a == 0 and b == 1: n01 += 1
        elif a == 1 and b == 0: n10 += 1
        else: n11 += 1
    if (n01 + n11) == 0 or (n00 + n01) == 0 or (n10 + n11) == 0:
        return np.nan, np.nan
    p01 = n01 / (n00 + n01)
    p11 = n11 / (n10 + n11)
    p = (n01 + n11) / (n00 + n01 + n10 + n11)
    if p <= 0 or p >= 1:
        return np.nan, np.nan

    def xlogy(cnt, prob):          # 0 * log(0) == 0
        return 0.0 if cnt == 0 else cnt * np.log(max(prob, 1e-300))

    ll0 = xlogy(n00 + n10, 1 - p) + xlogy(n01 + n11, p)
    ll1 = (xlogy(n00, 1 - p01) + xlogy(n01, p01)
           + xlogy(n10, 1 - p11) + xlogy(n11, p11))
    lr = -2 * (ll0 - ll1)
    return float(lr), float(1 - chi2.cdf(lr, 1))


# ============================================================================
# VaR-CONSTRAINED PORTFOLIO OPTIMIZER
# ============================================================================
#
# VaR-constrained portfolio construction.
#
# Formulation:
#
#     maximize    mu'w - (gamma/2) w'Sigma w - kappa ||w - w_prev||_1
#     subject to  sum(w) == 1        (long only)   or  sum(w) == 0 (market neutral)
#                 ||w||_1 <= gross
#                 |w_i| <= max_weight
#                 z_(1-a) * sqrt(w' Sigma_daily w) <= VaR budget
#
# The VaR constraint is a second-order cone constraint, which is why this is solved
# with a conic solver rather than SLSQP -- SLSQP will happily return a "solution"
# that violates it.
#
# The turnover term is inside the objective, not applied afterwards. Optimizing a
# portfolio and then regretting the trade list is not portfolio optimization.
#

import numpy as np
import cvxpy as cp
from scipy.stats import norm



def optimize(mu: np.ndarray, cov: np.ndarray, w_prev: np.ndarray,
             cfg: Config, var_budget: float) -> np.ndarray:
    n = mu.size
    valid = np.isfinite(mu) & np.isfinite(np.diag(cov))
    if valid.sum() < 5:
        return np.zeros(n)

    m = np.where(valid, mu, 0.0)
    S = cov.copy()
    S[~valid, :] = 0.0
    S[:, ~valid] = 0.0
    S[~valid, ~valid] = 1e-6
    ev, V = np.linalg.eigh((S + S.T) / 2)
    L = V @ np.diag(np.sqrt(np.clip(ev, 1e-14, None)))   # S = L L'

    z = norm.ppf(1 - cfg.var_alpha)
    w = cp.Variable(n)

    cons = [w[~valid] == 0,
            cp.abs(w) <= cfg.max_weight,
            cp.norm1(w) <= cfg.gross_leverage,
            cp.norm(L.T @ w, 2) <= var_budget / z]
    cons.append(cp.sum(w) == (1.0 if cfg.long_only else 0.0))
    if cfg.long_only:
        cons.append(w >= 0)

    obj = (m @ w
           - 0.5 * cfg.risk_aversion * cp.sum_squares(L.T @ w)
           - cfg.turnover_cost_kappa * cp.norm1(w - w_prev))

    prob = cp.Problem(cp.Maximize(obj), cons)
    for solver in (cp.CLARABEL, cp.SCS):
        try:
            prob.solve(solver=solver, verbose=False)
            if w.value is not None and np.isfinite(w.value).all():
                break
        except Exception:
            continue
    if w.value is None or not np.isfinite(w.value).all():
        return _fallback(m, np.diag(S), valid, cfg)

    out = np.asarray(w.value).ravel()
    out[~valid] = 0.0
    return out


def _fallback(mu, var, valid, cfg: Config) -> np.ndarray:
    """Inverse-vol weighted signal tilt. Used when the cone solve fails, so that a
    numerical hiccup produces a sane portfolio instead of a silent zero."""
    w = np.zeros_like(mu)
    s = np.where(valid, mu, 0.0)
    iv = np.where(valid & (var > 0), 1.0 / np.sqrt(np.maximum(var, 1e-12)), 0.0)
    raw = s * iv
    if not cfg.long_only:
        raw = raw - raw[valid].mean() * valid
    l1 = np.abs(raw).sum()
    if l1 > 1e-12:
        w = raw / l1 * cfg.gross_leverage
    return np.clip(w, -cfg.max_weight, cfg.max_weight)


def vol_target(w: np.ndarray, cov: np.ndarray, cfg: Config) -> np.ndarray:
    """Scale gross exposure so ex-ante annualized vol hits the target."""
    v = float(np.sqrt(max(w @ cov @ w, 1e-16)) * np.sqrt(252))
    if v < 1e-8:
        return w
    k = cfg.target_ann_vol / v
    k = min(k, cfg.max_gross_after_scaling / max(np.abs(w).sum(), 1e-9))
    return w * k


def trade_cost(w_new: np.ndarray, w_old: np.ndarray, dollar_vol: np.ndarray,
               capital: float, cfg: Config) -> float:
    """Spread + square-root market impact, as a fraction of capital.

    Impact is the term people leave out and then wonder why live results are half
    the backtest. A signal with 0.04 IC and 200% monthly turnover is a losing
    strategy no matter how elegant the formula that produced it.
    """
    dw = np.abs(w_new - w_old)
    spread = cfg.spread_bps / 1e4 * dw.sum()
    dv = np.where(np.isfinite(dollar_vol) & (dollar_vol > 0), dollar_vol, np.nan)
    part = np.where(np.isfinite(dv), dw * capital / np.maximum(dv, 1.0), 0.0)
    impact = cfg.impact_coef * np.sqrt(np.clip(part, 0, 1.0))
    return float(spread + (impact * dw).sum())


# ============================================================================
# PERFORMANCE + SIGNIFICANCE METRICS
# ============================================================================
#
# Performance and significance metrics.
#
# The Deflated Sharpe Ratio is the one that is not optional. If the search evaluated
# 80,000 expressions, the best of them will show a Sharpe near 2.0 on pure noise.
# Reporting a raw Sharpe from a selection process without disclosing the trial count
# is not a mistake, it is a category of self-deception with a name.
#

import numpy as np
from scipy.stats import norm, skew, kurtosis

EULER = 0.5772156649015329


def _clean(r):
    r = np.asarray(r, float)
    return r[np.isfinite(r)]


def sharpe(r, periods=252, rf=0.0):
    r = _clean(r) - rf / periods
    if r.size < 2 or r.std() == 0:
        return np.nan
    return float(r.mean() / r.std() * np.sqrt(periods))


def sortino(r, periods=252, mar=0.0):
    r = _clean(r) - mar / periods
    d = r[r < 0]
    dd = np.sqrt((d ** 2).mean()) if d.size else np.nan
    return float(r.mean() / dd * np.sqrt(periods)) if dd and dd > 0 else np.nan


def drawdown_curve(r):
    eq = np.cumprod(1 + _clean(r))
    peak = np.maximum.accumulate(eq)
    return eq / peak - 1.0


def max_drawdown(r):
    dd = drawdown_curve(r)
    return float(dd.min()) if dd.size else np.nan


def calmar(r, periods=252):
    r = _clean(r)
    if r.size < periods // 2:
        return np.nan
    ann = (1 + r).prod() ** (periods / r.size) - 1
    mdd = abs(max_drawdown(r))
    return float(ann / mdd) if mdd > 1e-9 else np.nan


def ann_return(r, periods=252):
    r = _clean(r)
    return float((1 + r).prod() ** (periods / max(r.size, 1)) - 1) if r.size else np.nan


def ann_vol(r, periods=252):
    r = _clean(r)
    return float(r.std() * np.sqrt(periods)) if r.size > 1 else np.nan


def hit_rate(r):
    r = _clean(r)
    return float((r > 0).mean()) if r.size else np.nan


def probabilistic_sharpe(r, sr_benchmark=0.0, periods=252):
    """P(true SR > benchmark), adjusted for skew and kurtosis of the returns."""
    r = _clean(r)
    n = r.size
    if n < 20:
        return np.nan
    sr = r.mean() / r.std()                       # per-period
    b = sr_benchmark / np.sqrt(periods)
    g3, g4 = skew(r), kurtosis(r, fisher=False)
    denom = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    return float(norm.cdf((sr - b) * np.sqrt(n - 1) / denom))


def deflated_sharpe(r, n_trials: int, trial_sr_var: float | None = None,
                    periods=252):
    """Bailey & Lopez de Prado (2014).

    trial_sr_var is the variance of the annualized Sharpe ratios across the trials
    of your search, measured over a sample of the SAME length as the reported
    track record. If not supplied, the null dispersion 252/T is used -- the
    variance of an estimated Sharpe when the true Sharpe is zero. That is the
    standard conservative default.

    Caveat worth stating out loud: n_trials counts unique expressions, but GP
    populations are full of near-duplicates, so the number of *independent* trials
    is smaller and this deflation is on the harsh side. It errs in the direction
    you want it to err.
    """
    r = _clean(r)
    if r.size < 20 or n_trials < 2:
        return np.nan
    v = trial_sr_var if trial_sr_var is not None else periods / r.size
    v = max(v, 1e-6)
    e1 = norm.ppf(1 - 1.0 / n_trials)
    e2 = norm.ppf(1 - 1.0 / (n_trials * np.e))
    sr0_ann = np.sqrt(v) * ((1 - EULER) * e1 + EULER * e2)
    return probabilistic_sharpe(r, sr_benchmark=sr0_ann, periods=periods)


def perf_summary(r, bench=None, n_trials=None, trial_sr_var=None,
            turnover=None, periods=252) -> dict:
    out = {
        "ann_return": ann_return(r, periods),
        "ann_vol": ann_vol(r, periods),
        "sharpe": sharpe(r, periods),
        "sortino": sortino(r, periods),
        "max_drawdown": max_drawdown(r),
        "calmar": calmar(r, periods),
        "hit_rate": hit_rate(r),
        "psr_vs_0": probabilistic_sharpe(r, 0.0, periods),
        "n_obs": int(_clean(r).size),
    }
    if n_trials:
        out["n_trials"] = int(n_trials)
        out["deflated_sharpe_prob"] = deflated_sharpe(r, n_trials, trial_sr_var, periods)
    if turnover is not None:
        out["avg_turnover_per_rebal"] = float(turnover)
    if bench is not None:
        b = _clean(bench)
        rr = _clean(r)
        k = min(b.size, rr.size)
        if k > 20:
            out["bench_sharpe"] = sharpe(b[:k], periods)
            act = rr[:k] - b[:k]
            out["information_ratio"] = sharpe(act, periods)
    return out


def fmt_metrics(d: dict) -> str:
    keys = ["ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown", "calmar",
            "hit_rate", "avg_turnover_per_rebal", "psr_vs_0", "n_trials",
            "deflated_sharpe_prob", "bench_sharpe", "information_ratio", "n_obs"]
    lines = []
    for k in keys:
        if k in d and d[k] is not None and np.isfinite(np.asarray(d[k], float)):
            v = d[k]
            s = f"{v:,.0f}" if k in ("n_trials", "n_obs") else f"{v:+.4f}"
            lines.append(f"  {k:<28s} {s}")
    return "\n".join(lines)


# ============================================================================
# PURGED WALK-FORWARD BACKTEST
# ============================================================================
#
# Purged, embargoed walk-forward backtest.
#
# The original spec's rule -- "never train on future data" -- is necessary but not
# sufficient. Two leaks survive a naive train/test split:
#
# PURGE. The last `horizon` days of the training window have labels that extend
# past the training window's end and into the test period. Those labels are future
# information. They must be dropped from training, not merely from testing.
#
# EMBARGO. Features on the first days of the test window are built from rolling
# windows that overlap the training data. Serial correlation carries information
# across the boundary. A gap of at least `horizon` days closes it.
#
# Both are cheap. Skipping them is the most common way a backtest with an
# apparently airtight design still reports a Sharpe it will not reproduce.
#

import numpy as np
import pandas as pd
from typing import Dict



def _slice_feats(feats: Dict[str, np.ndarray], sl) -> Dict[str, np.ndarray]:
    return {k: v[sl] for k, v in feats.items()}


def run_backtest(panel: dict, cfg: Config, verbose: bool = True) -> dict:
    feats = panel["features"]
    dates = panel["dates"]
    rets = panel["returns"]
    dvol = panel["dollar_vol"]
    vix = panel["vix"]
    label = panel["labels"]

    T, N = rets.shape
    fnames = list(feats)

    # first usable index: features need ~252d warmup
    warmup = 260
    starts = list(range(warmup + cfg.train_days,
                        T - cfg.test_days - cfg.horizon,
                        cfg.test_days))
    if not starts:
        raise ValueError("Not enough history for the requested walk-forward schedule.")

    port_ret_gross = np.full(T, np.nan)
    port_ret_net = np.full(T, np.nan)
    weights = np.zeros((T, N))
    var_pred = np.full(T, np.nan)
    var_pred_gbm = np.full(T, np.nan)
    turnovers, fold_log, all_ics = [], [], []
    n_trials_total = 0
    fold_sharpes = []

    w_prev = np.zeros(N)
    capital = 1.0

    for fi, test_start in enumerate(starts):
        tr_end = test_start - cfg.embargo_days           # embargo gap
        tr_start = tr_end - cfg.train_days
        purge_end = tr_end - cfg.horizon                 # purge overlapping labels
        te_end = min(test_start + cfg.test_days, T)

        tr = slice(tr_start, purge_end)
        n_tr = purge_end - tr_start
        val_len = int(n_tr * cfg.inner_val_frac)
        val_local = slice(n_tr - val_len, n_tr)

        f_tr = _slice_feats(feats, tr)
        y_tr = label[tr]

        reg = SymbolicRegressor(cfg, fnames,
                                rng=np.random.default_rng(cfg.seed + fi))
        survivors = reg.fit(f_tr, y_tr, verbose=False)
        n_trials_total += reg.n_evaluated

        trees, val_ics = select_candidates(survivors, reg, f_tr, y_tr, val_local, cfg)
        ic_hat = float(np.mean(val_ics)) if val_ics else 0.0

        # ---- apply to the test window ----
        te = slice(test_start, te_end)
        f_te = _slice_feats(feats, te)
        shape_te = (te_end - test_start, N)

        if trees:
            sig = combine_signals(trees, reg, f_te, shape_te)
            sig = orthogonalize(sig, f_te)
        else:
            sig = np.full(shape_te, np.nan)

        # realized IC on the out-of-sample window (diagnostic, not used for anything)
        ic_oos = row_corr(rank_rows(sig), label[te])
        all_ics.extend(ic_oos[np.isfinite(ic_oos)].tolist())

        fwd_vol = (pd.DataFrame(rets).rolling(60).std().to_numpy()[te]
                   * np.sqrt(cfg.horizon))
        mu_panel = to_expected_returns(sig, fwd_vol, ic_hat, cfg)

        fold_start_len = len(turnovers)
        for j, t in enumerate(range(test_start, te_end)):
            if j % cfg.rebalance_every == 0:
                hist = rets[max(0, t - cfg.cov_lookback):t]
                cov_h = shrunk_cov(hist, ewma_lambda=cfg.ewma_lambda)
                cov_d = cov_h                                   # daily units
                budget = (cfg.var_limit_stress if vix[t] >= cfg.vix_stress
                          else cfg.var_limit_daily)
                mu = mu_panel[j]
                mu = np.where(np.isfinite(mu), mu, np.nan)
                w = optimize(mu, cov_d, w_prev, cfg, budget)
                w = vol_target(w, cov_d, cfg)
                cost = trade_cost(w, w_prev, dvol[t], capital, cfg)
                turnovers.append(np.abs(w - w_prev).sum() / 2)
                w_prev = w

                # forward-looking risk numbers, recorded for later backtesting
                ph = hist @ np.where(np.isfinite(w), w, 0.0)
                fhs = filtered_historical_sim(ph, 1, cfg.fhs_paths, cfg.ewma_lambda,
                                              np.random.default_rng(cfg.seed + t))
                var_pred[t], _ = var_es(fhs, cfg.var_alpha)
                sd = float(np.sqrt(max(w @ cov_d @ w, 1e-16)))
                var_pred_gbm[t], _ = var_es(
                    gbm_sim(0.0, sd, 1, cfg.fhs_paths,
                            np.random.default_rng(cfg.seed + t)), cfg.var_alpha)
            else:
                cost = 0.0

            weights[t] = w_prev
            r_t = np.nan_to_num(rets[t], nan=0.0)
            g = float(w_prev @ r_t)
            port_ret_gross[t] = g
            port_ret_net[t] = g - cost
            capital *= (1 + port_ret_net[t])

        fold_r = port_ret_net[te]
        fold_r = fold_r[np.isfinite(fold_r)]
        if fold_r.size > 5 and fold_r.std() > 0:
            fold_sharpes.append(fold_r.mean() / fold_r.std() * np.sqrt(252))

        fold_log.append({
            "fold": fi,
            "train": f"{dates[tr_start].date()} -> {dates[purge_end - 1].date()}",
            "test": f"{dates[test_start].date()} -> {dates[te_end - 1].date()}",
            "n_members": len(trees),
            "val_ic": ic_hat,
            "oos_ic": float(np.nanmean(ic_oos)) if np.isfinite(ic_oos).any() else np.nan,
            "trials": reg.n_evaluated,
            "formulas": [to_str(t) for t in trees],
            "turnover": float(np.mean(turnovers[fold_start_len:]))
                        if len(turnovers) > fold_start_len else np.nan,
        })
        if verbose:
            fl = fold_log[-1]
            print(f"fold {fi:02d} | test {fl['test']} | members={fl['n_members']} "
                  f"| val_IC={fl['val_ic']:+.4f} | oos_IC={fl['oos_ic']:+.4f} "
                  f"| turn={fl['turnover']:.2%}")

    mask = np.isfinite(port_ret_net)
    return {
        "dates": dates,
        "gross": port_ret_gross,
        "net": port_ret_net,
        "mask": mask,
        "weights": weights,
        "var_pred_fhs": var_pred,
        "var_pred_gbm": var_pred_gbm,
        "turnover": float(np.mean(turnovers)) if turnovers else np.nan,
        "oos_ic_mean": float(np.mean(all_ics)) if all_ics else np.nan,
        "oos_ic_t": (float(np.mean(all_ics) / (np.std(all_ics) + 1e-12)
                           * np.sqrt(len(all_ics))) if len(all_ics) > 30 else np.nan),
        "n_trials": n_trials_total,
        "fold_sharpe_var": float(np.var(fold_sharpes)) if len(fold_sharpes) > 2 else None,
        "folds": fold_log,
        "tickers": panel["tickers"],
    }


# ============================================================================
# REPORTING
# ============================================================================
#
# Reporting and diagnostics.
#

import json
import os
import numpy as np
import pandas as pd



def build_report(res: dict, panel: dict, cfg: Config, outdir: str = "output") -> dict:
    os.makedirs(outdir, exist_ok=True)
    mask = res["mask"]
    dates = res["dates"][mask]
    net = res["net"][mask]
    gross = res["gross"][mask]
    bench = np.nan_to_num(panel["bench_ret"])[mask]

    stats_net = perf_summary(net, bench=bench, n_trials=res["n_trials"],
                          trial_sr_var=None,   # -> null dispersion 252/T; see metrics.deflated_sharpe
                          turnover=res["turnover"])
    stats_gross = perf_summary(gross)

    # ---- VaR model backtest ----
    vp = res["var_pred_fhs"][mask]
    have = np.isfinite(vp)
    exc = (net[have] < -vp[have])
    k_lr, k_p = kupiec_pof(exc, cfg.var_alpha)
    c_lr, c_p = christoffersen_independence(exc)
    var_report = {
        "expected_exceedance_rate": cfg.var_alpha,
        "observed_exceedance_rate": float(exc.mean()) if exc.size else np.nan,
        "kupiec_LR": k_lr, "kupiec_p": k_p,
        "christoffersen_LR": c_lr, "christoffersen_p": c_p,
        "mean_VaR_fhs": float(np.nanmean(res["var_pred_fhs"][mask])),
        "mean_VaR_gbm": float(np.nanmean(res["var_pred_gbm"][mask])),
    }
    var_report["gbm_understatement"] = (
        var_report["mean_VaR_gbm"] / var_report["mean_VaR_fhs"] - 1
        if var_report["mean_VaR_fhs"] else np.nan)

    # ---- artifacts ----
    eq = pd.DataFrame({
        "net": np.cumprod(1 + net),
        "gross": np.cumprod(1 + gross),
        "benchmark": np.cumprod(1 + bench),
        "drawdown": drawdown_curve(net),
        "ret_net": net,
        "var95_fhs": vp,
    }, index=dates)
    eq.to_csv(os.path.join(outdir, "equity_curve.csv"))

    W = pd.DataFrame(res["weights"][mask], index=dates, columns=res["tickers"])
    W.to_csv(os.path.join(outdir, "weights.csv"))

    with open(os.path.join(outdir, "formulas.json"), "w") as f:
        json.dump(res["folds"], f, indent=2)

    report = {"net": stats_net, "gross": stats_gross, "var_model": var_report,
              "oos_ic_mean": res["oos_ic_mean"], "oos_ic_t": res["oos_ic_t"]}
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)

    _charts(eq, W, outdir)
    return report


def _charts(eq: pd.DataFrame, W: pd.DataFrame, outdir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(3, 1, figsize=(11, 11), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1, 1]})
    ax[0].plot(eq.index, eq["gross"], lw=1, alpha=.6, label="strategy (gross)")
    ax[0].plot(eq.index, eq["net"], lw=1.4, label="strategy (net of costs)")
    ax[0].plot(eq.index, eq["benchmark"], lw=1, alpha=.7, label="benchmark")
    ax[0].set_yscale("log"); ax[0].legend(); ax[0].set_title("Cumulative growth of 1")
    ax[0].grid(alpha=.3)

    ax[1].fill_between(eq.index, eq["drawdown"], 0, alpha=.5)
    ax[1].set_title("Drawdown (net)"); ax[1].grid(alpha=.3)

    roll = eq["ret_net"].rolling(126)
    ax[2].plot(eq.index, roll.mean() / roll.std() * np.sqrt(252), lw=1)
    ax[2].axhline(0, color="k", lw=.8)
    ax[2].set_title("Rolling 6-month Sharpe (net)"); ax[2].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "performance.png"), dpi=120)
    plt.close(fig)

    mret = eq["ret_net"].resample("ME").apply(lambda s: (1 + s).prod() - 1)
    piv = pd.DataFrame({"y": mret.index.year, "m": mret.index.month, "r": mret.values})
    piv = piv.pivot(index="y", columns="m", values="r")
    fig, a = plt.subplots(figsize=(10, max(2.5, 0.45 * len(piv))))
    im = a.imshow(piv.values, cmap="RdYlGn", aspect="auto",
                  vmin=-np.nanmax(np.abs(piv.values)), vmax=np.nanmax(np.abs(piv.values)))
    a.set_xticks(range(piv.shape[1])); a.set_xticklabels(piv.columns)
    a.set_yticks(range(piv.shape[0])); a.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                a.text(j, i, f"{v*100:.1f}", ha="center", va="center", fontsize=7)
    a.set_title("Monthly returns, % (net)"); fig.colorbar(im, ax=a)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "monthly_heatmap.png"), dpi=120)
    plt.close(fig)


def to_text(report: dict, res: dict) -> str:
    L = []
    L.append("NET OF COSTS"); L.append(fmt_metrics(report["net"]))
    L.append("\nGROSS (for reference only -- you cannot trade this)")
    L.append(fmt_metrics(report["gross"]))
    L.append("\nSIGNAL")
    L.append(f"  {'oos mean daily IC':<28s} {report['oos_ic_mean']:+.4f}")
    L.append(f"  {'oos IC t-stat':<28s} {report['oos_ic_t']:+.2f}")
    L.append("\nVaR MODEL BACKTEST (95%, 1-day)")
    v = report["var_model"]
    L.append(f"  {'expected exceedance rate':<28s} {v['expected_exceedance_rate']:.3f}")
    L.append(f"  {'observed exceedance rate':<28s} {v['observed_exceedance_rate']:.3f}")
    L.append(f"  {'Kupiec p-value':<28s} {v['kupiec_p']:.3f}   (low = wrong coverage)")
    L.append(f"  {'Christoffersen p-value':<28s} {v['christoffersen_p']:.3f}   (low = clustered breaches)")
    L.append(f"  {'mean VaR, filtered hist.':<28s} {v['mean_VaR_fhs']:.4f}")
    L.append(f"  {'mean VaR, GBM Monte Carlo':<28s} {v['mean_VaR_gbm']:.4f}")
    L.append(f"  {'GBM error vs FHS':<28s} {v['gbm_understatement']:+.1%}")
    return "\n".join(L)


# ============================================================================
# CLI
# ============================================================================

def self_test(seed: int = 3) -> None:
    """Harness validation. These are the checks that matter.

    A backtest that has never been run on data with no alpha in it is an untested
    backtest. If the null check fails, every number the system produces on real
    data is uninterpretable -- fix the harness before looking at a single result.
    """
    def _run(planted):
        cfg = Config(seed=seed, pop_size=150, generations=6)
        raw = synthetic_panel(n_days=1300, n_names=40, seed=seed, planted_ic=planted)
        panel = build_features(raw, cfg)
        panel["labels"] = make_labels(panel["returns"], cfg)
        return run_backtest(panel, cfg, verbose=False)

    print("[1/4] label construction contains no lookahead ...", end=" ", flush=True)
    cfg = Config()
    r = np.random.default_rng(0).normal(0, .01, (300, 10))
    lab = make_labels(r, cfg)
    assert np.isnan(lab[-cfg.horizon:]).all()
    print("ok")

    print("[2/4] VaR constraint is respected by the optimizer ...", end=" ", flush=True)
    c2 = Config(max_weight=0.5, gross_leverage=2.0)
    rng = np.random.default_rng(1)
    A = rng.normal(size=(20, 12)) * 0.01
    cov = A.T @ A / 20 + np.eye(12) * 1e-4
    w = optimize(rng.normal(0, 0.01, 12), cov, np.zeros(12), c2, var_budget=0.01)
    assert norm.ppf(0.95) * np.sqrt(w @ cov @ w) <= 0.01 + 1e-6
    print("ok")

    print("[3/4] null panel: pipeline must find nothing ...", end=" ", flush=True)
    res = _run(0.0)
    net = res["net"][res["mask"]]
    assert abs(res["oos_ic_mean"]) < 0.02, "found alpha in noise -- the harness leaks"
    assert deflated_sharpe(net, res["n_trials"]) < 0.9
    print(f"ok (oos IC {res['oos_ic_mean']:+.4f}, Sharpe {sharpe(net):+.2f})")

    print("[4/4] planted alpha: pipeline must find it ...", end=" ", flush=True)
    res = _run(0.06)
    net = res["net"][res["mask"]]
    assert res["oos_ic_mean"] > 0.02, "cannot recover a signal that is definitely there"
    assert sharpe(net) > 0.8
    print(f"ok (oos IC {res['oos_ic_mean']:+.4f}, Sharpe {sharpe(net):+.2f})")
    print("\nall checks passed.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--real", action="store_true", help="yfinance download")
    g.add_argument("--synthetic-null", action="store_true", help="no alpha; must find nothing")
    g.add_argument("--synthetic-alpha", action="store_true", help="planted alpha; must find it")
    g.add_argument("--self-test", action="store_true", help="run harness assertions")
    ap.add_argument("--outdir", default="output")
    ap.add_argument("--pop", type=int, default=None, help="GP population size")
    ap.add_argument("--gens", type=int, default=None, help="GP generations")
    ap.add_argument("--long-only", action="store_true",
                    help="default is dollar-neutral long/short")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    if a.self_test:
        self_test(a.seed)
        return

    cfg = Config(seed=a.seed, long_only=a.long_only)
    if a.pop:
        cfg.pop_size = a.pop
    if a.gens:
        cfg.generations = a.gens

    if a.real:
        raw = download_data(cfg)
    else:
        raw = synthetic_panel(seed=a.seed,
                              planted_ic=0.0 if a.synthetic_null else 0.06)

    panel = build_features(raw, cfg)
    panel["labels"] = make_labels(panel["returns"], cfg)
    print(f"panel: {panel['returns'].shape[0]} days x {panel['returns'].shape[1]} "
          f"names, {len(panel['features'])} features\n")

    res = run_backtest(panel, cfg)
    rep = build_report(res, panel, cfg, outdir=a.outdir)
    print("\n" + "=" * 68)
    print(to_text(rep, res))
    print("=" * 68)
    print(f"\nartifacts -> {a.outdir}/  "
          "(equity_curve.csv, weights.csv, formulas.json, summary.json, *.png)")


if __name__ == "__main__":
    main()
