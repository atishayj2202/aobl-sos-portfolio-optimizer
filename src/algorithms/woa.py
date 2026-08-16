import numpy as np
import random

def WOA(obj_func, pop: np.ndarray, map_func, iters: int, 
        lb: float = None, ub: float = None, is_portfolio: bool = False):
    """
    Whale Optimization Algorithm (WOA).
    """
    pop = pop.copy()
    pop_size, dim = pop.shape
    fitness = np.array([obj_func(ind) for ind in pop], dtype=float)
    
    best_idx = np.argmin(fitness)
    leader_pos = pop[best_idx].copy()
    leader_score = fitness[best_idx]
    
    best_curve = []
    b = 1  # spiral constant
    
    for t in range(iters):
        a = 2.0 - t * (2.0 / iters)    # Decreases linearly from 2 to 0
        a2 = -1.0 + t * (-1.0 / iters) # Decreases linearly from -1 to -2
        
        for i in range(pop_size):
            r1 = random.random()
            r2 = random.random()
            
            A = 2 * a * r1 - a
            C = 2 * r2
            
            p = random.random()
            l = (a2 - 1) * random.random() + 1
            
            if p < 0.5:
                if abs(A) < 1:
                    D = abs(C * leader_pos - pop[i])
                    pop[i] = leader_pos - A * D
                else:
                    rand_idx = random.randrange(pop_size)
                    rand_pos = pop[rand_idx]
                    D = abs(C * rand_pos - pop[i])
                    pop[i] = rand_pos - A * D
            else:
                distance_to_leader = abs(leader_pos - pop[i])
                pop[i] = distance_to_leader * np.exp(b * l) * np.cos(2 * np.pi * l) + leader_pos
                
            pop[i] = map_func(pop[i])
            fit = obj_func(pop[i])
            fitness[i] = fit
            
            if fit < leader_score:
                leader_score = fit
                leader_pos = pop[i].copy()
                
        best_curve.append(float(leader_score))
        
    return float(leader_score), leader_pos, best_curve
