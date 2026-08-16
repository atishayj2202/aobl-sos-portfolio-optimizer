# Adaptive Opposition-Based Learning Symbiotic Organisms Search for Robust Portfolio Optimization

This repository contains the supplementary source code, datasets, and result matrices for the manuscript submitted to *Swarm and Evolutionary Computation (SWEVO)*.

## 1. Overview
Standard Swarm Intelligence algorithms frequently stagnate in high-dimensional financial optimization spaces. This repository implements **Adaptive Opposition-Based Learning Symbiotic Organisms Search (AOBL-SOS)**, proposing:
1. **Rank-Reversal Simplex Opposition**: A mathematically robust mechanism that preserves budget ($\sum w_i = 1$) and strict regulatory caps ($w_i \le 20\%$) without destructive normalization.
2. **Stagnation-Triggered Adaptive OBL**: An injection operator that bypasses local optima, evaluated on 8 benchmark functions (up to D=200).



## 2. Mathematical Benchmark Validation
AOBL-SOS demonstrates near-perfect exploitation on unimodal landscapes, achieving **$7.46 \times 10^{-39}$** fitness on the Sphere function (D=30). Extensive statistical validation via Friedman Rank Test ($p < 0.0001$, Rank 2.43) confirms active superiority across the 8 multimodal baselines.

## 3. Repository Structure

*   `data/`: S&P 500 daily price matrix used for validation.
*   `figures/`: Convergence charts and visualization outputs.
*   `results/`: Raw CSV data validating the manuscript tables, scalability analysis, and non-parametric Nemenyi/Friedman tests.
*   `src/`: 
    *   `algorithms/`: Implementations of AOBL-SOS, SOS, PSO, GWO, HHO, WOA, SCA, DE.
    *   `benchmarks/`: Objective functions (Ackley, Rastrigin, Sphere, etc.).
    *   `portfolio/`: Data fetching, constraints capping, and evaluation logic.
*   `run_benchmarks.py`: Main execution script to replicate benchmark studies.
*   `run_portfolio.py`: Main execution script to replicate the financial portfolio experiments.

## 4. How to Replicate

```bash
# 1. Install dependencies
pip install numpy pandas yfinance scipy matplotlib

# 2. Run Mathematical Benchmarks
python run_benchmarks.py

# 3. Run Out-Of-Sample Portfolio Optimizer
python run_portfolio.py
```