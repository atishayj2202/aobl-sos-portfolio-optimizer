import numpy as np
import pandas as pd
from src.utils import normalize_weights, normalize_and_cap

def sharpe_objective(w: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf_daily: float) -> float:
    """Compute the negative annualized Sharpe Ratio for minimization."""
    w = normalize_weights(w)
    ret = np.dot(mu, w)
    var = np.dot(w.T, np.dot(cov, w))
    if var <= 0:
        return 1e6
    ann_ret = (ret - rf_daily) * 252
    ann_vol = np.sqrt(var) * np.sqrt(252)
    if ann_vol <= 0:
        return 1e6
    return -ann_ret / ann_vol

def evaluate_portfolio(weights: np.ndarray, returns: pd.DataFrame, 
                       rf_annual: float = 0.05, cap: float = 0.05) -> dict:
    """Calculate annualized returns, volatility, Sharpe, Sortino, and drawdown metrics."""
    w = normalize_and_cap(weights, cap)
    # Using matrix multiplication on underlying values
    port_ret = returns.values @ w
    
    ann_ret = np.mean(port_ret) * 252
    ann_vol = np.std(port_ret) * np.sqrt(252)
    
    # Sharpe using annual risk-free rate
    sharpe = (ann_ret - rf_annual) / ann_vol if ann_vol > 0 else 0.0
    
    # Sortino using downside deviation
    neg_ret = port_ret[port_ret < 0]
    down_vol = np.std(neg_ret) * np.sqrt(252) if len(neg_ret) > 0 else 1e-6
    sortino = (ann_ret - rf_annual) / down_vol if down_vol > 0 else 0.0
    
    # Maximum Drawdown calculation
    cum_returns = np.cumprod(1 + port_ret)
    peak = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - peak) / peak
    max_dd = drawdowns.min()
    
    return {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "daily_returns": port_ret,
        "weights": w
    }

def equal_weight_benchmark(returns: pd.DataFrame, rf_annual: float = 0.05, 
                           cap: float = 0.05) -> dict:
    """Evaluate an equal-weighted portfolio as a benchmark."""
    n = returns.shape[1]
    w = np.ones(n) / n
    return evaluate_portfolio(w, returns, rf_annual, cap)
