import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.utils import set_seed, normalize_and_cap
from src.algorithms.aobl_sos import AOBL_SOS
from src.algorithms.sos import SOS
from src.portfolio.data_v2 import load_data_v2 as load_data
from src.portfolio.evaluation import evaluate_portfolio, equal_weight_benchmark

def min_variance_portfolio(cov, cap=0.20):
    n = cov.shape[0]
    def obj(w):
        return np.dot(w.T, np.dot(cov, w))
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = [(0, cap) for _ in range(n)]
    init_w = np.ones(n) / n
    res = minimize(obj, init_w, method='SLSQP', bounds=bounds, constraints=cons)
    return res.x if res.success else init_w

def max_sharpe_portfolio(mu, cov, rf_daily, cap=0.20):
    n = len(mu)
    def obj(w):
        ret = np.dot(w, mu)
        vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        return -(ret - rf_daily) / (vol + 1e-12)
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = [(0, cap) for _ in range(n)]
    init_w = np.ones(n) / n
    res = minimize(obj, init_w, method='SLSQP', bounds=bounds, constraints=cons)
    return res.x if res.success else init_w

def inverse_volatility_portfolio(cov, cap=0.20):
    vols = np.sqrt(np.diag(cov))
    inv_vols = 1.0 / (vols + 1e-12)
    w = inv_vols / np.sum(inv_vols)
    return normalize_and_cap(w, cap)

def compute_net_metrics(weights, test_ret_df, cost_bps=10, rebal_freq=21, rf_annual=0.05, cap=0.20):
    """
    Computes out-of-sample portfolio performance taking into account rebalancing turnover and transaction costs.
    cost_bps: cost in basis points (e.g. 5, 10, 15)
    rebal_freq: trading days between rebalances (21 = monthly, 63 = quarterly, 126 = semi-annual)
    """
    weights = normalize_and_cap(weights, cap)
    n_days, n_assets = test_ret_df.shape
    c = cost_bps / 10000.0 # convert bps to fraction
    
    current_w = weights.copy()
    daily_net_returns = []
    turnover_list = []
    
    for t in range(n_days):
        day_ret = test_ret_df.iloc[t].values
        # Gross return on day t
        gross_r = np.dot(current_w, day_ret)
        
        # Check rebalance
        if t > 0 and t % rebal_freq == 0:
            target_w = weights.copy()
            turnover = np.sum(np.abs(target_w - current_w))
            cost = turnover * c
            net_r = gross_r - cost
            turnover_list.append(turnover)
            current_w = target_w.copy()
        else:
            net_r = gross_r
            # Update drifted weights
            current_w = current_w * (1 + day_ret)
            if np.sum(current_w) > 0:
                current_w = current_w / np.sum(current_w)
                
        daily_net_returns.append(net_r)
        
    daily_net_returns = np.array(daily_net_returns)
    cum_ret = np.cumprod(1 + daily_net_returns)
    ann_return = np.mean(daily_net_returns) * 252
    ann_vol = np.std(daily_net_returns, ddof=1) * np.sqrt(252)
    sharpe = (ann_return - rf_annual) / (ann_vol + 1e-12)
    
    downside = daily_net_returns[daily_net_returns < 0]
    downside_std = np.std(downside, ddof=1) * np.sqrt(252) if len(downside) > 0 else 1e-12
    sortino = (ann_return - rf_annual) / downside_std
    
    peaks = np.maximum.accumulate(cum_ret)
    drawdowns = (peaks - cum_ret) / peaks
    max_dd = np.max(drawdowns)
    
    avg_turnover = np.mean(turnover_list) if len(turnover_list) > 0 else 0.0
    
    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "avg_turnover": avg_turnover,
        "cum_returns": cum_ret,
        "daily_returns": daily_net_returns
    }

def run_jpm_experiments(output_dir="jpm/v2"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"==================================================")
    print(f"  RUNNING JPM PRACTITIONER EXPERIMENTS & ANALYSIS")
    print(f"==================================================")
    
    set_seed(42)
    tickers, mu, cov, train_ret, test_ret, source = load_data(n_stocks=179, seed=42)
    cap = 0.20
    rf_annual = 0.02
    rf_daily = rf_annual / 252
    
    # 1. Optimize AOBL-SOS, SOS, PSO, DE weights
    print("\n[1] Optimizing Portfolios for JPM Analysis...")
    dim = len(mu)
    obj = lambda w: - (np.dot(w, mu) - rf_daily) / (np.sqrt(np.dot(w.T, np.dot(cov, w))) + 1e-12)
    map_func = lambda w: normalize_and_cap(w, cap)
    
    pop = np.random.uniform(0.0, 1.0, (50, dim))
    pop = np.array([map_func(p) for p in pop])
    
    print("  Running AOBL-SOS...")
    _, w_aobl, _ = AOBL_SOS(obj, pop.copy(), map_func, iters=500, is_portfolio=True, init_obl=True, obl_mode="portfolio_reversal", cap=cap)
    
    print("  Running Standard SOS...")
    _, w_sos, _ = SOS(obj, pop.copy(), map_func, iters=500, is_portfolio=True)
    
    print("  Computing Min Variance...")
    w_minvar = min_variance_portfolio(cov, cap=cap)
    
    print("  Computing Max Sharpe (Markowitz)...")
    w_maxsh = max_sharpe_portfolio(mu, cov, rf_daily, cap=cap)
    
    print("  Computing Inverse Volatility (Risk Parity)...")
    w_invvol = inverse_volatility_portfolio(cov, cap=cap)
    
    w_eq = np.ones(dim) / dim
    
    # 2. Evaluate Transaction Costs (0 bps, 5 bps, 10 bps, 15 bps)
    print("\n[2] Transaction Cost Sensitivity Analysis (0, 5, 10, 15 bps)...")
    cost_levels = [0, 5, 10, 15]
    tx_results = []
    
    portfolios = {
        "AOBL-SOS (Proposed)": w_aobl,
        "SOS (Baseline)": w_sos,
        "Max Sharpe (Markowitz)": w_maxsh,
        "Min Variance": w_minvar,
        "Risk Parity (Inv-Vol)": w_invvol,
        "Equal Weight (1/N)": w_eq
    }
    
    for name, w in portfolios.items():
        for cost in cost_levels:
            res = compute_net_metrics(w, test_ret, cost_bps=cost, rebal_freq=21, rf_annual=rf_annual, cap=cap)
            tx_results.append({
                "Portfolio": name,
                "Cost_bps": cost,
                "Net_Ann_Return_Pct": res["ann_return"] * 100,
                "Net_Ann_Vol_Pct": res["ann_vol"] * 100,
                "Net_Sharpe": res["sharpe"],
                "Net_Sortino": res["sortino"],
                "Net_Max_Drawdown_Pct": res["max_drawdown"] * 100,
                "Avg_Turnover_Pct": res["avg_turnover"] * 100
            })
            
    df_tx = pd.DataFrame(tx_results)
    df_tx.to_csv(os.path.join(output_dir, "transaction_cost_sensitivity.csv"), index=False)
    print(f"Saved transaction cost sensitivity to {output_dir}/transaction_cost_sensitivity.csv")
    
    # 3. Master Portfolio Benchmark Comparison Table (at 10 bps cost)
    df_10bps = df_tx[df_tx["Cost_bps"] == 10].sort_values("Net_Sharpe", ascending=False)
    df_10bps.to_csv(os.path.join(output_dir, "jpm_master_benchmark_table_10bps.csv"), index=False)
    print("\nMaster Benchmark Table (at 10 bps Transaction Cost):")
    print(df_10bps.to_string(index=False))
    
    # 4. Rebalancing Frequency Analysis (Monthly = 21d, Quarterly = 63d, Semi-Annual = 126d)
    print("\n[3] Rebalancing Frequency Analysis...")
    rebal_results = []
    freq_map = {"Monthly (21d)": 21, "Quarterly (63d)": 63, "Semi-Annual (126d)": 126}
    
    for freq_name, freq_days in freq_map.items():
        for name, w in portfolios.items():
            res = compute_net_metrics(w, test_ret, cost_bps=10, rebal_freq=freq_days, rf_annual=rf_annual, cap=cap)
            rebal_results.append({
                "Rebalance_Freq": freq_name,
                "Portfolio": name,
                "Net_Sharpe": res["sharpe"],
                "Net_Sortino": res["sortino"],
                "Net_Ann_Return_Pct": res["ann_return"] * 100,
                "Net_Max_Drawdown_Pct": res["max_drawdown"] * 100,
                "Avg_Turnover_Pct": res["avg_turnover"] * 100
            })
            
    df_rebal = pd.DataFrame(rebal_results)
    df_rebal.to_csv(os.path.join(output_dir, "rebalancing_frequency_analysis.csv"), index=False)
    print(f"Saved rebalancing frequency analysis to {output_dir}/rebalancing_frequency_analysis.csv")

    # 5. Walk-Forward Expanding Window Analysis
    print("\n[4] Walk-Forward Expanding Window Validation...")
    # Partition dataset into 5 expanding windows
    n_total = len(train_ret) + len(test_ret)
    full_ret = pd.concat([train_ret, test_ret], axis=0).reset_index(drop=True)
    
    wf_results = []
    # Windows: Train 2012-2017 (1500d) -> Test 2018 (250d), etc.
    splits = [
        ("2012-2017 -> 2018", 1250, 1500),
        ("2012-2018 -> 2019", 1500, 1750),
        ("2012-2019 -> 2020 (COVID)", 1750, 2000),
        ("2012-2020 -> 2021", 2000, 2250),
        ("2012-2021 -> 2022 (Hikes)", 2250, 2500),
        ("2012-2022 -> 2023-2025", 2500, len(full_ret))
    ]
    
    for split_name, train_end, test_end in splits:
        tr = full_ret.iloc[:train_end]
        te = full_ret.iloc[train_end:test_end]
        
        m_tr = tr.mean().values * 252
        c_tr = tr.cov().values * 252
        
        # Optimize AOBL-SOS on this window
        obj_w = lambda w: - (np.dot(w, m_tr) - rf_annual) / (np.sqrt(np.dot(w.T, np.dot(c_tr, w))) + 1e-12)
        set_seed(42)
        pop_w = np.random.uniform(0.0, 1.0, (30, dim))
        pop_w = np.array([map_func(p) for p in pop_w])
        _, w_wf_aobl, _ = AOBL_SOS(obj_w, pop_w, map_func, iters=200, is_portfolio=True, init_obl=True, obl_mode="portfolio_reversal", cap=cap)
        _, w_wf_sos, _ = SOS(obj_w, pop_w, map_func, iters=200, is_portfolio=True)
        
        w_wf_eq = np.ones(dim) / dim
        w_wf_maxsh = max_sharpe_portfolio(m_tr, c_tr, rf_annual/252, cap=cap)
        w_wf_minvar = min_variance_portfolio(c_tr, cap=cap)
        
        res_aobl = compute_net_metrics(w_wf_aobl, te, cost_bps=10, rebal_freq=21, rf_annual=rf_annual, cap=cap)
        res_sos = compute_net_metrics(w_wf_sos, te, cost_bps=10, rebal_freq=21, rf_annual=rf_annual, cap=cap)
        res_eq = compute_net_metrics(w_wf_eq, te, cost_bps=10, rebal_freq=21, rf_annual=rf_annual, cap=cap)
        res_maxsh = compute_net_metrics(w_wf_maxsh, te, cost_bps=10, rebal_freq=21, rf_annual=rf_annual, cap=cap)
        res_minvar = compute_net_metrics(w_wf_minvar, te, cost_bps=10, rebal_freq=21, rf_annual=rf_annual, cap=cap)
        
        wf_results.append({
            "Window": split_name,
            "AOBL_SOS_Sharpe": res_aobl["sharpe"],
            "SOS_Sharpe": res_sos["sharpe"],
            "Markowitz_Sharpe": res_maxsh["sharpe"],
            "MinVar_Sharpe": res_minvar["sharpe"],
            "EqualWeight_Sharpe": res_eq["sharpe"],
            "AOBL_SOS_NetReturn_Pct": res_aobl["ann_return"] * 100,
            "AOBL_SOS_MaxDD_Pct": res_aobl["max_drawdown"] * 100
        })
        
    df_wf = pd.DataFrame(wf_results)
    df_wf.to_csv(os.path.join(output_dir, "walk_forward_validation.csv"), index=False)
    print(f"Saved walk-forward validation to {output_dir}/walk_forward_validation.csv")
    print(df_wf.to_string(index=False))

    # 6. Generate Figures for JPM Paper
    print("\n[5] Generating High-Resolution Figures for JPM Paper...")
    
    # Figure 1: Cumulative Net-of-Cost Portfolio Performance (10 bps cost)
    plt.figure(figsize=(12, 6))
    res_aobl = compute_net_metrics(w_aobl, test_ret, cost_bps=10, rebal_freq=21, rf_annual=rf_annual, cap=cap)
    res_sos = compute_net_metrics(w_sos, test_ret, cost_bps=10, rebal_freq=21, rf_annual=rf_annual, cap=cap)
    res_maxsh = compute_net_metrics(w_maxsh, test_ret, cost_bps=10, rebal_freq=21, rf_annual=rf_annual, cap=cap)
    res_minvar = compute_net_metrics(w_minvar, test_ret, cost_bps=10, rebal_freq=21, rf_annual=rf_annual, cap=cap)
    res_invvol = compute_net_metrics(w_invvol, test_ret, cost_bps=10, rebal_freq=21, rf_annual=rf_annual, cap=cap)
    res_eq = compute_net_metrics(w_eq, test_ret, cost_bps=10, rebal_freq=21, rf_annual=rf_annual, cap=cap)
    
    plt.plot(res_aobl["cum_returns"], color='#D85A30', lw=2.5, label='AOBL-SOS (Proposed)')
    plt.plot(res_sos["cum_returns"], color='#378ADD', lw=2.0, label='SOS (Baseline)')
    plt.plot(res_maxsh["cum_returns"], color='#9C27B0', lw=1.8, label='Markowitz Max Sharpe')
    plt.plot(res_minvar["cum_returns"], color='#FF9800', lw=1.8, label='Minimum Variance')
    plt.plot(res_invvol["cum_returns"], color='#009688', lw=1.8, label='Risk Parity (Inv-Vol)')
    plt.plot(res_eq["cum_returns"], color='#6D4C41', lw=1.8, linestyle='--', label='Equal-Weight (1/N)')
    
    plt.title('Out-of-Sample Cumulative Net Return (10 bps Transaction Cost & Monthly Rebalancing)', fontsize=13, fontweight='bold')
    plt.xlabel('Trading Days (Out-of-Sample Test Period: 2023–2025)')
    plt.ylabel('Net Portfolio Growth (Base = 1.0)')
    plt.legend(loc='upper left', frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig1_path = os.path.join(output_dir, "jpm_fig1_net_cumulative_returns.png")
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    print(f"  Saved Figure 1 -> {fig1_path}")
    
    # Figure 2: Net Sharpe Ratio vs. Transaction Costs (Sensitivity)
    plt.figure(figsize=(10, 5))
    for name in portfolios.keys():
        sub = df_tx[df_tx["Portfolio"] == name]
        plt.plot(sub["Cost_bps"], sub["Net_Sharpe"], marker='o', lw=2, label=name)
    plt.xlabel('Transaction Cost (basis points per trade)')
    plt.ylabel('Net Sharpe Ratio')
    plt.title('Sensitivity of Net Sharpe Ratio to Transaction Costs')
    plt.xticks([0, 5, 10, 15])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig2_path = os.path.join(output_dir, "jpm_fig2_cost_sensitivity.png")
    plt.savefig(fig2_path, dpi=200)
    plt.close()
    print(f"  Saved Figure 2 -> {fig2_path}")
    
    print("\n==================================================")
    print("  JPM PRACTITIONER EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    run_jpm_experiments()
