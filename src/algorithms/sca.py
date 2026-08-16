import numpy as np
import random

def SCA(obj_func, pop: np.ndarray, map_func, iters: int, 
        lb: float = None, ub: float = None, is_portfolio: bool = False):
    """
    Sine Cosine Algorithm (SCA).
    """
    pop = pop.copy()
    pop_size, dim = pop.shape
    fitness = np.array([obj_func(ind) for ind in pop], dtype=float)
    
    best_idx = np.argmin(fitness)
    dest_pos = pop[best_idx].copy()
    dest_score = fitness[best_idx]
    
    best_curve = []
    a = 2.0  # Constant
    
    for t in range(iters):
        r1 = a - t * (a / iters)  # r1 decreases linearly from a to 0
        
        for i in range(pop_size):
            for j in range(dim):
                r2 = 2 * np.pi * random.random()
                r3 = 2 * random.random()
                r4 = random.random()
                
                if r4 < 0.5:
                    pop[i, j] = pop[i, j] + r1 * np.sin(r2) * abs(r3 * dest_pos[j] - pop[i, j])
                else:
                    pop[i, j] = pop[i, j] + r1 * np.cos(r2) * abs(r3 * dest_pos[j] - pop[i, j])
                    
            pop[i] = map_func(pop[i])
            fit = obj_func(pop[i])
            fitness[i] = fit
            
            if fit < dest_score:
                dest_score = fit
                dest_pos = pop[i].copy()
                
        best_curve.append(float(dest_score))
        
    return float(dest_score), dest_pos, best_curve
