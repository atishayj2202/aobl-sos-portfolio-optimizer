import random
import numpy as np
from src.algorithms.sos import mutualism_step, commensalism_step, parasitism_step
from src.algorithms.obl import (
    classic_opposition,
    quasi_opposition,
    portfolio_classic_opposition,
    portfolio_reversal_opposition
)

def apply_obl_replacement(pop: np.ndarray, fitness: np.ndarray, obj_func, 
                          map_func, replace_frac: float, mode: str, 
                          lb: float = None, ub: float = None, cap: float = 0.05):
    """Replace worst fraction of population with their OBL counterpart if they perform better."""
    n = len(pop)
    k = max(1, int(n * replace_frac))
    worst_idx = np.argsort(fitness)[-k:]
    
    if mode == "classic":
        opp = classic_opposition(pop[worst_idx], lb, ub)
    elif mode == "quasi":
        opp = quasi_opposition(pop[worst_idx], lb, ub)
    elif mode == "portfolio_classic":
        opp = portfolio_classic_opposition(pop[worst_idx], cap)
    elif mode == "portfolio_reversal":
        opp = portfolio_reversal_opposition(pop[worst_idx])
    else:
        raise ValueError(f"Unknown OBL mode: {mode}")
        
    # Ensure they satisfy constraints
    if mode in ["classic", "quasi"]:
        opp = np.clip(opp, lb, ub)
        
    opp_fit = np.array([obj_func(ind) for ind in opp], dtype=float)
    improved = opp_fit < fitness[worst_idx]
    
    pop[worst_idx[improved]] = opp[improved]
    fitness[worst_idx[improved]] = opp_fit[improved]
    return pop, fitness

def AOBL_SOS(
    obj_func,
    pop: np.ndarray,
    map_func,
    iters: int,
    lb: float = None,
    ub: float = None,
    is_portfolio: bool = False,
    init_obl: bool = True,
    obl_mode: str = "quasi",          # "classic", "quasi", "portfolio_classic", "portfolio_reversal"
    replace_frac: float = 0.5,
    patience: int = 15,
    eps: float = 1e-12,
    p0: float = 0.20,
    pmax: float = 0.95,
    cap: float = 0.05
):
    """
    Adaptive Opposition-Based Learning Symbiotic Organisms Search (AOBL-SOS).
    """
    pop = pop.copy()
    fitness = np.array([obj_func(ind) for ind in pop], dtype=float)
    
    # --- Initial OBL ---
    if init_obl:
        if obl_mode == "classic":
            opp = classic_opposition(pop, lb, ub)
        elif obl_mode == "quasi":
            opp = quasi_opposition(pop, lb, ub)
        elif obl_mode == "portfolio_classic":
            opp = portfolio_classic_opposition(pop, cap)
        elif obl_mode == "portfolio_reversal":
            opp = portfolio_reversal_opposition(pop)
        else:
            raise ValueError(f"Unknown OBL mode: {obl_mode}")
            
        if obl_mode in ["classic", "quasi"]:
            opp = np.clip(opp, lb, ub)
            
        combined = np.vstack([pop, opp])
        combined_fit = np.array([obj_func(ind) for ind in combined], dtype=float)
        idx = np.argsort(combined_fit)[:len(pop)]
        pop = combined[idx]
        fitness = combined_fit[idx]
        
    best_curve = []
    best_val = float(np.min(fitness))
    stagnation = 0
    
    for t in range(iters):
        best = pop[np.argmin(fitness)]
        
        # --- Standard SOS Phases ---
        for i in range(len(pop)):
            pop, fitness = mutualism_step(pop, fitness, best, obj_func, map_func, i)
            best = pop[np.argmin(fitness)]
            pop, fitness = commensalism_step(pop, fitness, best, obj_func, map_func, i)
            best = pop[np.argmin(fitness)]
            pop, fitness = parasitism_step(pop, fitness, obj_func, map_func, i, 
                                           lb=lb, ub=ub, is_portfolio=is_portfolio)
            
        current_best = float(np.min(fitness))
        best_curve.append(current_best)
        
        # --- Stagnation tracking ---
        if current_best < best_val - eps:
            best_val = current_best
            stagnation = 0
        else:
            stagnation += 1
            
        # --- Adaptive OBL Trigger ---
        if stagnation >= patience:
            p = min(pmax, p0 + 0.05 * (stagnation - patience + 1))
            if random.random() < p:
                pop, fitness = apply_obl_replacement(
                    pop, fitness, obj_func, map_func,
                    replace_frac=replace_frac,
                    mode=obl_mode,
                    lb=lb, ub=ub, cap=cap
                )
                stagnation = max(0, patience // 2)
                
    idx = np.argmin(fitness)
    return float(fitness[idx]), pop[idx], best_curve
