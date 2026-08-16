import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.utils import set_seed, map_to_bounds, normalize_and_cap
from src.algorithms.sos import SOS
from src.algorithms.aobl_sos import AOBL_SOS
from src.algorithms.pso import PSO
from src.algorithms.de import DE
from src.algorithms.gwo import GWO
from src.algorithms.woa import WOA
from src.algorithms.hho import HHO
from src.algorithms.sca import SCA

from src.benchmarks.functions import BENCHMARK_FUNCTIONS
from src.portfolio.data_v2 import load_data_v2 as load_data
from src.portfolio.evaluation import sharpe_objective, evaluate_portfolio, equal_weight_benchmark

def compute_nemenyi_pvalues(matrix_data):
    """
    Computes pairwise Nemenyi post-hoc p-values for matrix of shape (n_benchmarks, n_algos).
    """
    N, k = matrix_data.shape
    ranks = np.zeros_like(matrix_data)
    for i in range(N):
        ranks[i] = pd.Series(matrix_data[i]).rank().values
    mean_ranks = np.mean(ranks, axis=0)
    
    se = np.sqrt(k * (k + 1) / (6.0 * N))
    p_values = np.zeros((k, k))
    
    for i in range(k):
        for j in range(k):
            if i == j:
                p_values[i, j] = 1.0
            else:
                z = abs(mean_ranks[i] - mean_ranks[j]) / se
                p_values[i, j] = 2.0 * (1.0 - norm.cdf(z))
                
    return mean_ranks, p_values

def run_swevo_experiments(output_dir="swevo/v3"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"==================================================", flush=True)
    print(f"  RUNNING SWEVO EXPANDED BENCHMARK EXPERIMENTS", flush=True)
    print(f"==================================================", flush=True)
    
    algos = {
        "AOBL-SOS": AOBL_SOS,
        "SOS": SOS,
        "PSO": PSO,
        "DE": DE,
        "GWO": GWO,
        "WOA": WOA,
        "HHO": HHO,
        "SCA": SCA
    }
    
    benchmark_names = ["Sphere", "Rastrigin", "Rosenbrock", "Ackley", "Griewank", "Schwefel", "Zakharov", "Levy"]
    runs = 5
    iters = 100
    pop_size = 50
    dim = 30
    
    results_summary = []
    algo_ranks_data = {name: [] for name in algos.keys()}
    execution_times = {name: [] for name in algos.keys()}
    
    # 1. Run Benchmark Evaluations
    for b_name in benchmark_names:
        b_info = BENCHMARK_FUNCTIONS[b_name]
        func = b_info["func"]
        lb, ub = b_info["bounds"]
        map_func = lambda x: map_to_bounds(x, lb, ub)
        
        print(f"\nEvaluating Benchmark: {b_name} (D={dim})...", flush=True)
        
        b_results = {}
        for algo_name, algo_fn in algos.items():
            run_vals = []
            start_time = time.time()
            
            for r in range(runs):
                set_seed(42 + r)
                pop = np.random.uniform(lb, ub, (pop_size, dim))
                
                if algo_name == "AOBL-SOS":
                    val, _, _ = algo_fn(func, pop.copy(), map_func, iters=iters, lb=lb, ub=ub,
                                        init_obl=True, obl_mode="quasi", patience=15)
                else:
                    val, _, _ = algo_fn(func, pop.copy(), map_func, iters=iters, lb=lb, ub=ub)
                    
                run_vals.append(val)
                
            elapsed = (time.time() - start_time) / runs
            execution_times[algo_name].append(elapsed)
            
            run_arr = np.array(run_vals)
            b_results[algo_name] = {
                "mean": run_arr.mean(),
                "std": run_arr.std(),
                "best": run_arr.min(),
                "worst": run_arr.max()
            }
            
            print(f"  {algo_name:<10}: Mean={run_arr.mean():.4e} ± {run_arr.std():.4e}", flush=True)
            
        for idx, a in enumerate(algos.keys()):
            algo_ranks_data[a].append(b_results[a]["mean"])
            
        for a in algos.keys():
            results_summary.append({
                "Benchmark": b_name,
                "Algorithm": a,
                "Mean": b_results[a]["mean"],
                "Std": b_results[a]["std"],
                "Best": b_results[a]["best"],
                "Worst": b_results[a]["worst"],
                "Avg_Time_Sec": np.mean(execution_times[a])
            })
            
    df_summary = pd.DataFrame(results_summary)
    df_summary.to_csv(os.path.join(output_dir, "benchmark_8algos_results.csv"), index=False)
    print(f"\nSaved benchmark results to {output_dir}/benchmark_8algos_results.csv", flush=True)
    
    # 2. Compute Friedman Test & Nemenyi Post-hoc Test
    print("\nComputing Friedman Rank Test across 8 algorithms...", flush=True)
    matrix_data = np.array([algo_ranks_data[a] for a in algos.keys()]).T
    
    stat, p_val = friedmanchisquare(*matrix_data.T)
    print(f"Friedman Test Stat (Chi-square): {stat:.4f}, p-value: {p_val:.4e}", flush=True)
    
    avg_ranks, nemenyi_pvals = compute_nemenyi_pvalues(matrix_data)
    
    rank_df = pd.DataFrame({
        "Algorithm": list(algos.keys()),
        "Average_Rank": avg_ranks
    }).sort_values("Average_Rank")
    
    print("\nAlgorithm Ranks (Lower is better):", flush=True)
    print(rank_df.to_string(index=False), flush=True)
    rank_df.to_csv(os.path.join(output_dir, "friedman_ranks.csv"), index=False)
    
    df_nemenyi = pd.DataFrame(nemenyi_pvals, index=list(algos.keys()), columns=list(algos.keys()))
    df_nemenyi.to_csv(os.path.join(output_dir, "nemenyi_pvalues.csv"))
    print(f"Saved Friedman ranks and Nemenyi p-values to {output_dir}/", flush=True)

    # Plot Ranks
    plt.figure(figsize=(10, 5))
    plt.barh(rank_df["Algorithm"][::-1], rank_df["Average_Rank"][::-1], color="#378ADD")
    plt.xlabel("Average Friedman Rank (Lower is better)")
    plt.title("Friedman Test Ranking Across 8 Benchmark Functions")
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "friedman_rank_chart.png"), dpi=150)
    plt.close()
    
    # 3. Scalability Analysis (Dimensions 30, 50, 100, 200)
    print("\nRunning Scalability Analysis across D in {30, 50, 100, 200}...", flush=True)
    dims = [30, 50, 100, 200]
    scalability_results = []
    
    for D in dims:
        for b_name in ["Sphere", "Rastrigin", "Rosenbrock", "Ackley"]:
            b_info = BENCHMARK_FUNCTIONS[b_name]
            func = b_info["func"]
            lb, ub = b_info["bounds"]
            map_func = lambda x: map_to_bounds(x, lb, ub)
            
            for algo_name, algo_fn in [("AOBL-SOS", AOBL_SOS), ("SOS", SOS), ("PSO", PSO), ("DE", DE), ("GWO", GWO)]:
                run_vals = []
                for r in range(3):
                    set_seed(100 + r)
                    pop = np.random.uniform(lb, ub, (pop_size, D))
                    if algo_name == "AOBL-SOS":
                        val, _, _ = algo_fn(func, pop.copy(), map_func, iters=50, lb=lb, ub=ub, init_obl=True, obl_mode="quasi")
                    else:
                        val, _, _ = algo_fn(func, pop.copy(), map_func, iters=50, lb=lb, ub=ub)
                    run_vals.append(val)
                mean_val = float(np.mean(run_vals))
                scalability_results.append({
                    "Dimension": D,
                    "Benchmark": b_name,
                    "Algorithm": algo_name,
                    "Mean_Fitness": mean_val
                })
                
    df_scalability = pd.DataFrame(scalability_results)
    df_scalability.to_csv(os.path.join(output_dir, "scalability_analysis.csv"), index=False)
    print(f"Saved scalability analysis to {output_dir}/scalability_analysis.csv", flush=True)

    # 4. Ablation Sensitivity Analysis
    print("\nRunning Ablation Sensitivity Analysis...", flush=True)
    patience_vals = [5, 10, 15, 20]
    frac_vals = [0.2, 0.3, 0.5, 0.7]
    ablation_results = []
    
    func = BENCHMARK_FUNCTIONS["Rastrigin"]["func"]
    lb, ub = BENCHMARK_FUNCTIONS["Rastrigin"]["bounds"]
    map_func = lambda x: map_to_bounds(x, lb, ub)
    
    for tau in patience_vals:
        for frac in frac_vals:
            run_vals = []
            for r in range(3):
                set_seed(200 + r)
                pop = np.random.uniform(lb, ub, (pop_size, 30))
                val, _, _ = AOBL_SOS(func, pop.copy(), map_func, iters=50, lb=lb, ub=ub,
                                     init_obl=True, obl_mode="quasi", patience=tau, replace_frac=frac)
                run_vals.append(val)
            ablation_results.append({
                "Patience_Tau": tau,
                "Replace_Frac": frac,
                "Mean_Fitness": float(np.mean(run_vals))
            })
            
    df_ablation = pd.DataFrame(ablation_results)
    df_ablation.to_csv(os.path.join(output_dir, "ablation_sensitivity.csv"), index=False)
    print(f"Saved ablation sensitivity to {output_dir}/ablation_sensitivity.csv", flush=True)

    # 5. Out-of-Sample Portfolio Optimization with All 8 Algorithms
    print("\nEvaluating Portfolio Optimization across All 8 Algorithms...", flush=True)
    tickers, mu, cov, train_ret, test_ret, source = load_data(n_stocks=179, seed=42)
    rf_annual = 0.02
    rf_daily = rf_annual / 252
    cap = 0.20
    obj = lambda w: sharpe_objective(w, mu, cov, rf_daily)
    map_func = lambda w: normalize_and_cap(w, cap)
    
    port_results = []
    set_seed(42)
    pop_base = np.random.uniform(0.0, 1.0, (pop_size, len(mu)))
    pop_base = np.array([map_func(p) for p in pop_base])

    for algo_name, algo_fn in algos.items():
        if algo_name == "AOBL-SOS":
            v, w, _ = algo_fn(obj, pop_base.copy(), map_func, iters=500, is_portfolio=True, init_obl=True, obl_mode="quasi", lb=0.0, ub=1.0, cap=cap)
        else:
            v, w, _ = algo_fn(obj, pop_base.copy(), map_func, iters=500, is_portfolio=True)
            
        metrics = evaluate_portfolio(w, test_ret, rf_annual, cap)
        port_results.append({
            "Algorithm": algo_name,
            "Ann_Return_Pct": metrics["ann_return"] * 100,
            "Ann_Volatility_Pct": metrics["ann_vol"] * 100,
            "Sharpe_Ratio": metrics["sharpe"],
            "Sortino_Ratio": metrics["sortino"],
            "Max_Drawdown_Pct": metrics["max_drawdown"] * 100
        })
        
    eq_metrics = equal_weight_benchmark(test_ret, rf_annual, cap)
    port_results.append({
        "Algorithm": "Equal-Weight",
        "Ann_Return_Pct": eq_metrics["ann_return"] * 100,
        "Ann_Volatility_Pct": eq_metrics["ann_vol"] * 100,
        "Sharpe_Ratio": eq_metrics["sharpe"],
        "Sortino_Ratio": eq_metrics["sortino"],
        "Max_Drawdown_Pct": eq_metrics["max_drawdown"] * 100
    })
    
    df_port = pd.DataFrame(port_results)
    df_port.to_csv(os.path.join(output_dir, "portfolio_8algos_out_of_sample.csv"), index=False)
    print(f"\nSaved 8-algorithm portfolio results to {output_dir}/portfolio_8algos_out_of_sample.csv", flush=True)
    print(df_port.to_string(index=False), flush=True)
    
    print("\n==================================================", flush=True)
    print("  SWEVO EXPANDED EXPERIMENTS COMPLETED SUCCESSFULLY!", flush=True)
    print("==================================================", flush=True)

if __name__ == "__main__":
    run_swevo_experiments()
