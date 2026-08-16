import numpy as np

def GWO(obj_func, pop: np.ndarray, map_func, iters: int, 
        lb: float = None, ub: float = None, is_portfolio: bool = False):
    """
    Grey Wolf Optimizer (GWO).
    """
    pop = pop.copy()
    pop_size, dim = pop.shape
    fitness = np.array([obj_func(ind) for ind in pop], dtype=float)
    
    # Sort initial population to identify Alpha, Beta, Delta wolves
    idx = np.argsort(fitness)
    alpha_pos = pop[idx[0]].copy()
    alpha_score = fitness[idx[0]]
    
    beta_pos = pop[idx[1]].copy()
    beta_score = fitness[idx[1]]
    
    delta_pos = pop[idx[2]].copy()
    delta_score = fitness[idx[2]]
    
    best_curve = []
    
    for t in range(iters):
        a = 2.0 - t * (2.0 / iters)  # Decreases linearly from 2 to 0
        
        for i in range(pop_size):
            for j in range(dim):
                # Alpha update
                r1, r2 = np.random.random(), np.random.random()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha_pos[j] - pop[i, j])
                X1 = alpha_pos[j] - A1 * D_alpha
                
                # Beta update
                r1, r2 = np.random.random(), np.random.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta_pos[j] - pop[i, j])
                X2 = beta_pos[j] - A2 * D_beta
                
                # Delta update
                r1, r2 = np.random.random(), np.random.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta_pos[j] - pop[i, j])
                X3 = delta_pos[j] - A3 * D_delta
                
                # Average position update
                pop[i, j] = (X1 + X2 + X3) / 3.0
                
            # Apply mapping/constraints
            pop[i] = map_func(pop[i])
            fit = obj_func(pop[i])
            fitness[i] = fit
            
            # Update Alpha, Beta, Delta
            if fit < alpha_score:
                delta_score = beta_score
                delta_pos = beta_pos.copy()
                beta_score = alpha_score
                beta_pos = alpha_pos.copy()
                alpha_score = fit
                alpha_pos = pop[i].copy()
            elif fit < beta_score:
                delta_score = beta_score
                delta_pos = beta_pos.copy()
                beta_score = fit
                beta_pos = pop[i].copy()
            elif fit < delta_score:
                delta_score = fit
                delta_pos = pop[i].copy()
                
        best_curve.append(float(alpha_score))
        
    return float(alpha_score), alpha_pos, best_curve
