import os
import sys
import numpy as np

# Adjust path to import src modules properly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.algorithms.aobl_sos import AOBL_SOS
from src.benchmarks.functions import sphere

def test_benchmark():
    print("--- Testing AOBL-SOS on Sphere Function (Anti-P-Hacking Check) ---")
    lb, ub = -100, 100
    optimizer = AOBL_SOS(obj_func=sphere, lb=lb, ub=ub, pop_size=20, max_iter=100, stagnation_patience=5)
    best_sol, best_fit, curve = optimizer.optimize()
    print(f"Final Best Fitness (Sphere): {best_fit:.5e}")
    if best_fit < 1e-10:
        print("✅ Mathematical Exploitation Confirmed: Algorithm successfully converges towards global optimum.")
    else:
        print("❌ Warning: Convergence suboptimal.")

def test_portfolio():
    print("\n--- Testing Portfolio Optimization Mechanics ---")
    # Generate dummy covariance matrix and expected returns
    np.random.seed(42)
    n_assets = 10
    mu = np.random.normal(0.05, 0.02, n_assets)
    # create positive semi-definite cov matrix
    A = np.random.randn(n_assets, n_assets)
    cov = np.dot(A, A.transpose()) * 0.001
    
    rf = 0.02
    
    # Portfolio Objective: Minimize Negative Sharpe
    def negative_sharpe(w):
        ret = np.dot(w, mu)
        vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        if vol == 0: return float('inf')
        return -(ret - rf) / vol

    lb = np.zeros(n_assets)
    ub = np.ones(n_assets) * 0.20 # 20% cap

    optimizer = AOBL_SOS(obj_func=negative_sharpe, lb=lb, ub=ub, pop_size=20, max_iter=100, stagnation_patience=5)
    best_w, best_fit, _ = optimizer.optimize()
    
    print(f"Best Found Negative Sharpe: {best_fit:.5f} (Sharpe Ratio: {-best_fit:.5f})")
    print(f"Weight Sum: {np.sum(best_w):.5f}")
    print(f"Max Weight: {np.max(best_w):.5f}")
    
    if abs(np.sum(best_w) - 1.0) < 1e-3 and np.max(best_w) <= 0.20 + 1e-3:
        print("✅ Constraints Confirmed: Weights sum to 1 and respect the 20% cardinality cap.")
    else:
        print("❌ Constraint Violation Detected!")

if __name__ == "__main__":
    test_benchmark()
    test_portfolio()
