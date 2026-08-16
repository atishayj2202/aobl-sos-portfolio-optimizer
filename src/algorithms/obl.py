import numpy as np
from src.utils import normalize_and_cap

def classic_opposition(pop: np.ndarray, lb: float, ub: float) -> np.ndarray:
    """Classic OBL: Reflect across the hypercube center."""
    return lb + ub - pop

def quasi_opposition(pop: np.ndarray, lb: float, ub: float) -> np.ndarray:
    """Quasi-OBL: Sample between the midpoint and the classic opposite."""
    m = (lb + ub) / 2.0
    x_op = lb + ub - pop
    r = np.random.uniform(0.0, 1.0, size=pop.shape)
    x_q = m + r * (x_op - m)
    return np.clip(x_q, lb, ub)

def portfolio_classic_opposition(pop: np.ndarray, cap: float = 0.05) -> np.ndarray:
    """
    Classic OBL adapted to portfolio simplex space.
    Computes 1 - w and then normalizes and caps the result.
    """
    opp = 1.0 - pop
    if len(opp.shape) == 1:
        return normalize_and_cap(opp, cap)
    return np.array([normalize_and_cap(ind, cap) for ind in opp])

def portfolio_reversal_opposition(pop: np.ndarray) -> np.ndarray:
    """
    A rank-reversal opposition operator for portfolio simplex spaces.
    Swaps the highest allocations with the lowest allocations.
    Since it is a permutation of the original weights, it automatically
    satisfies the sum-to-one and cap constraints without renormalization.
    """
    if len(pop.shape) == 1:
        sort_idx = np.argsort(pop)
        opp = np.zeros_like(pop)
        opp[sort_idx] = pop[sort_idx[::-1]]
        return opp
    
    opp_pop = np.zeros_like(pop)
    for idx, ind in enumerate(pop):
        sort_idx = np.argsort(ind)
        opp_pop[idx][sort_idx] = ind[sort_idx[::-1]]
    return opp_pop
