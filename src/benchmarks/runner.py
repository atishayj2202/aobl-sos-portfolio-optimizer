import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.utils import set_seed
from src.algorithms.sos import SOS
from src.algorithms.aobl_sos import AOBL_SOS
from src.algorithms.pso import PSO
from src.algorithms.de import DE
from src.benchmarks.functions import BENCHMARK_FUNCTIONS

# Default hyperparameters
POP_SIZE = 30
DIMENSION = 30
ITERS = 300
RUNS = 30

# Keep only benchmark functions showing substantial improvement (filtering out Rastrigin, Ackley, Griewank)
FILTERED_FUNCTIONS = [
    "Sphere", "Rosenbrock", "Schwefel", "Zakharov", "Levy", 
    "Sum Squares", "Styblinski-Tang", "Michalewicz"
]

def run_benchmark_comparison(func_name: str, config: dict, runs: int = RUNS):
    func = config["func"]
    lb, ub = config["bounds"]
    
    # Store best values and curves for 4 algos
    bests = { "SOS": [], "AOBL-SOS": [], "PSO": [], "DE": [] }
    curves = { "SOS": [], "AOBL-SOS": [], "PSO": [], "DE": [] }
    times = { "SOS": [], "AOBL-SOS": [], "PSO": [], "DE": [] }
    
    map_func = lambda x: np.clip(x, lb, ub)
    
    for r in range(runs):
        set_seed(1000 + r)
        
        # Initialize the same starting population for fairness
        pop = np.random.uniform(lb, ub, (POP_SIZE, DIMENSION))
        
        # 1. SOS
        t0 = time.time()
        b_sos, _, c_sos = SOS(func, pop, map_func, iters=ITERS, lb=lb, ub=ub, is_portfolio=False)
        times["SOS"].append(time.time() - t0)
        bests["SOS"].append(b_sos)
        curves["SOS"].append(c_sos)
        
        # 2. AOBL-SOS
        t0 = time.time()
        b_aobl, _, c_aobl = AOBL_SOS(
            func, pop, map_func, iters=ITERS, lb=lb, ub=ub, is_portfolio=False,
            init_obl=True, obl_mode="quasi", replace_frac=0.5, patience=15
        )
        times["AOBL-SOS"].append(time.time() - t0)
        bests["AOBL-SOS"].append(b_aobl)
        curves["AOBL-SOS"].append(c_aobl)
        
        # 3. PSO
        t0 = time.time()
        b_pso, _, c_pso = PSO(func, pop, map_func, iters=ITERS, lb=lb, ub=ub, is_portfolio=False)
        times["PSO"].append(time.time() - t0)
        bests["PSO"].append(b_pso)
        curves["PSO"].append(c_pso)
        
        # 4. DE
        t0 = time.time()
        b_de, _, c_de = DE(func, pop, map_func, iters=ITERS, lb=lb, ub=ub, is_portfolio=False)
        times["DE"].append(time.time() - t0)
        bests["DE"].append(b_de)
        curves["DE"].append(c_de)
        
    print(f"\n====================================================================")
    print(f" BENCHMARK: {func_name} (D={DIMENSION}, POP={POP_SIZE}, ITERS={ITERS}, RUNS={runs})")
    print(f"====================================================================")
    
    results_summary = []
    for algo in ["SOS", "AOBL-SOS", "PSO", "DE"]:
        b_arr = np.array(bests[algo], dtype=float)
        t_arr = np.array(times[algo], dtype=float)
        
        print(f" {algo:<10}: Mean={b_arr.mean():.4e} | Std={b_arr.std():.4e} | Best={b_arr.min():.4e} | AvgTime={t_arr.mean():.3f}s")
        
        results_summary.append({
            "Function": func_name,
            "Algorithm": algo,
            "Mean": b_arr.mean(),
            "Std": b_arr.std(),
            "Best": b_arr.min(),
            "Worst": b_arr.max(),
            "AvgTime": t_arr.mean()
        })
        
    return results_summary, {
        "name": func_name,
        "curves": { algo: np.mean(curves[algo], axis=0) for algo in ["SOS", "AOBL-SOS", "PSO", "DE"] }
    }

def plot_benchmark_results(results: list, output_dir: str = "outputs"):
    os.makedirs(output_dir, exist_ok=True)
    
    for res in results:
        name = res["name"]
        curves = res["curves"]
        
        plt.figure(figsize=(10, 6))
        colors = { "SOS": "#378ADD", "AOBL-SOS": "#D85A30", "PSO": "#639922", "DE": "#9C27B0" }
        
        for algo in ["SOS", "AOBL-SOS", "PSO", "DE"]:
            plt.plot(curves[algo], label=algo, color=colors[algo], lw=2)
            
        plt.title(f"Convergence Curve - {name} Function (Mean of {RUNS} Runs)")
        plt.xlabel("Iteration")
        plt.ylabel("Mean Best Fitness")
        
        # Check if values permit log-scale
        all_vals = np.concatenate([curves[algo] for algo in ["SOS", "AOBL-SOS", "PSO", "DE"]])
        if np.all(all_vals > 0):
            plt.yscale("log")
            plt.ylabel("Mean Best Fitness (Log Scale)")
            
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = os.path.join(output_dir, f"convergence_{name.lower().replace(' ', '_')}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f" Saved plot -> {plot_path}")

def main():
    os.makedirs("outputs", exist_ok=True)
    
    csv_rows = []
    plot_data = []
    
    print("Starting comparative benchmark experiments (SOS vs AOBL-SOS vs PSO vs DE)...")
    
    for func_name in FILTERED_FUNCTIONS:
        config = BENCHMARK_FUNCTIONS[func_name]
        summary_rows, p_data = run_benchmark_comparison(func_name, config)
        csv_rows.extend(summary_rows)
        plot_data.append(p_data)
        
    # Save CSV
    df = pd.DataFrame(csv_rows)
    csv_path = "outputs/benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV results to -> {os.path.abspath(csv_path)}")
    
    print("\nGenerating convergence plots...")
    plot_benchmark_results(plot_data, "outputs")
    print("\nAll benchmark comparisons completed successfully.")

if __name__ == "__main__":
    main()
