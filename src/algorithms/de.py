import random
import numpy as np

def DE(obj_func, pop: np.ndarray, map_func, iters: int, 
       lb: float = None, ub: float = None, is_portfolio: bool = False):
    """
    Standard Differential Evolution (DE/rand/1/bin) optimization.
    """
    pop = pop.copy()
    pop_size, dim = pop.shape
    fitness = np.array([obj_func(ind) for ind in pop], dtype=float)
    
    # Parameters
    F = 0.8   # Mutation scaling factor
    CR = 0.9  # Crossover rate
    
    best_curve = []
    
    for t in range(iters):
        for i in range(pop_size):
            # Select three random individuals distinct from i
            candidates = list(range(pop_size))
            candidates.remove(i)
            r1, r2, r3 = random.sample(candidates, 3)
            
            # Mutation
            mutant = pop[r1] + F * (pop[r2] - pop[r3])
            
            # Crossover
            trial = pop[i].copy()
            j_rand = random.randrange(dim)
            for j in range(dim):
                if random.random() < CR or j == j_rand:
                    trial[j] = mutant[j]
                    
            # Map trial to feasible space
            trial = map_func(trial)
            
            # Selection
            trial_fit = obj_func(trial)
            if trial_fit < fitness[i]:
                pop[i] = trial
                fitness[i] = trial_fit
                
        best_curve.append(float(np.min(fitness)))
        
    idx = np.argmin(fitness)
    return float(fitness[idx]), pop[idx], best_curve
