import numpy as np
import random
import math

def HHO(obj_func, pop: np.ndarray, map_func, iters: int, 
        lb: float = None, ub: float = None, is_portfolio: bool = False):
    """
    Harris Hawks Optimization (HHO).
    """
    pop = pop.copy()
    pop_size, dim = pop.shape
    fitness = np.array([obj_func(ind) for ind in pop], dtype=float)
    
    rabbit_idx = np.argmin(fitness)
    rabbit_pos = pop[rabbit_idx].copy()
    rabbit_score = fitness[rabbit_idx]
    
    best_curve = []
    
    for t in range(iters):
        E0 = 2 * random.random() - 1  # Initial energy in [-1, 1]
        E = 2 * E0 * (1 - (t / iters)) # Escaping energy
        
        for i in range(pop_size):
            if abs(E) >= 1: # Exploration phase
                q = random.random()
                rand_hawk_idx = random.randrange(pop_size)
                X_rand = pop[rand_hawk_idx]
                if q >= 0.5:
                    pop[i] = X_rand - random.random() * abs(X_rand - 2 * random.random() * pop[i])
                else:
                    X_mean = np.mean(pop, axis=0)
                    if lb is not None and ub is not None and not is_portfolio:
                        pop[i] = (rabbit_pos - X_mean) - random.random() * ((ub - lb) * random.random() + lb)
                    else:
                        pop[i] = (rabbit_pos - X_mean) - random.random() * (rabbit_pos - pop[i])
            else: # Exploitation phase
                r = random.random()
                if r >= 0.5 and abs(E) >= 0.5:
                    # Soft besiege
                    pop[i] = (rabbit_pos - pop[i]) - E * abs(2 * (1 - random.random()) * rabbit_pos - pop[i])
                elif r >= 0.5 and abs(E) < 0.5:
                    # Hard besiege
                    pop[i] = rabbit_pos - E * abs(rabbit_pos - pop[i])
                elif r < 0.5 and abs(E) >= 0.5:
                    # Soft besiege with progressive rapid dives
                    J = 2 * (1 - random.random())
                    Y = rabbit_pos - E * abs(J * rabbit_pos - pop[i])
                    Y = map_func(Y)
                    if obj_func(Y) < fitness[i]:
                        pop[i] = Y
                    else:
                        S = np.random.uniform(-1, 1, dim)
                        Z = Y + S * np.random.randn(dim)
                        Z = map_func(Z)
                        if obj_func(Z) < fitness[i]:
                            pop[i] = Z
                elif r < 0.5 and abs(E) < 0.5:
                    # Hard besiege with progressive rapid dives
                    J = 2 * (1 - random.random())
                    X_mean = np.mean(pop, axis=0)
                    Y = rabbit_pos - E * abs(J * rabbit_pos - X_mean)
                    Y = map_func(Y)
                    if obj_func(Y) < fitness[i]:
                        pop[i] = Y
                    else:
                        S = np.random.uniform(-1, 1, dim)
                        Z = Y + S * np.random.randn(dim)
                        Z = map_func(Z)
                        if obj_func(Z) < fitness[i]:
                            pop[i] = Z
                            
            pop[i] = map_func(pop[i])
            fit = obj_func(pop[i])
            fitness[i] = fit
            
            if fit < rabbit_score:
                rabbit_score = fit
                rabbit_pos = pop[i].copy()
                
        best_curve.append(float(rabbit_score))
        
    return float(rabbit_score), rabbit_pos, best_curve
