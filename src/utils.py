import random
import numpy as np

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

def map_to_bounds(x: np.ndarray, lb: float, ub: float) -> np.ndarray:
    """Clip candidate solutions within search space bounds [lb, ub]."""
    return np.clip(x, lb, ub)

def normalize_weights(w: np.ndarray) -> np.ndarray:
    """Normalize weights to sum to 1 with non-negative values."""
    w = np.maximum(w, 0.0)
    s = np.sum(w)
    return w / s if s > 0 else np.ones_like(w) / len(w)

def apply_cap(w: np.ndarray, cap: float) -> np.ndarray:
    """
    Enforce per-asset allocation cap iteratively.
    Any weight exceeding `cap` is clipped and the excess
    is redistributed proportionally to uncapped assets.
    """
    w = np.maximum(w, 0.0)
    for _ in range(200):  # Max redistribution passes
        excess = np.maximum(w - cap, 0.0)
        if excess.sum() < 1e-10:
            break
        w = np.minimum(w, cap)
        uncapped_mask = w < cap
        uncapped_sum = w[uncapped_mask].sum()
        if uncapped_sum > 1e-10:
            w[uncapped_mask] += excess.sum() * (w[uncapped_mask] / uncapped_sum)
        else:
            # All assets already at cap — spread equally
            w[:] = cap
    s = w.sum()
    return w / s if s > 0 else np.ones_like(w) / len(w)

def normalize_and_cap(w: np.ndarray, cap: float = 0.05) -> np.ndarray:
    """Normalize weights and then enforce the per-asset allocation cap."""
    return apply_cap(normalize_weights(w), cap)
