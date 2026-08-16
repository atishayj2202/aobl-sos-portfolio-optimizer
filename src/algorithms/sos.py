import random
import numpy as np

def mutualism_step(pop: np.ndarray, fitness: np.ndarray, best: np.ndarray, 
                   obj_func, map_func, i: int):
    """Mutualism step where i and a random organism j interact cooperatively."""
    n = len(pop)
    j = random.randrange(n)
    while j == i:
        j = random.randrange(n)
        
    mutual_vector = (pop[i] + pop[j]) / 2.0
    BF1 = random.choice([1, 2])
    BF2 = random.choice([1, 2])
    
    xi_new = map_func(pop[i] + random.random() * (best - BF1 * mutual_vector))
    xj_new = map_func(pop[j] + random.random() * (best - BF2 * mutual_vector))
    
    f_xi = obj_func(xi_new)
    if f_xi < fitness[i]:
        pop[i] = xi_new
        fitness[i] = f_xi
        
    f_xj = obj_func(xj_new)
    if f_xj < fitness[j]:
        pop[j] = xj_new
        fitness[j] = f_xj
        
    return pop, fitness

def commensalism_step(pop: np.ndarray, fitness: np.ndarray, best: np.ndarray, 
                      obj_func, map_func, i: int):
                      
    """Commensalism step where i benefits from best without affecting random j."""
    n = len(pop)
    j = random.randrange(n)
    while j == i:
        j = random.randrange(n)
        
    xi_new = map_func(pop[i] + random.uniform(-1, 1) * (best - pop[j]))
    
    f_xi = obj_func(xi_new)
    if f_xi < fitness[i]:
        pop[i] = xi_new
        fitness[i] = f_xi
        
    return pop, fitness

def parasitism_step(pop: np.ndarray, fitness: np.ndarray, obj_func, map_func, i: int,
                    lb: float = None, ub: float = None, is_portfolio: bool = False):
    """
    Parasitism step where a mutated parasite from i tries to replace a random host j.
    If is_portfolio=True, we sample the mutation from [0, 1] before mapping.
    Otherwise, we sample within [lb, ub].
    """
    n = len(pop)
    j = random.randrange(n)
    while j == i:
        j = random.randrange(n)
        
    parasite = pop[i].copy()
    dim = len(parasite)
    
    # Mutate 10% of the dimensions
    k = max(1, dim // 10)
    idxs = random.sample(range(dim), k)
    
    for idx in idxs:
        if is_portfolio:
            parasite[idx] = random.random()
        else:
            parasite[idx] = random.uniform(lb, ub)
            
    parasite = map_func(parasite)
    
    f_p = obj_func(parasite)
    if f_p < fitness[j]:
        pop[j] = parasite
        fitness[j] = f_p
        
    return pop, fitness

def SOS(obj_func, pop: np.ndarray, map_func, iters: int, 
        lb: float = None, ub: float = None, is_portfolio: bool = False):
    """
    Standard Symbiotic Organisms Search (SOS).
    """
    pop = pop.copy()
    fitness = np.array([obj_func(ind) for ind in pop], dtype=float)
    best_curve = []
    
    for t in range(iters):
        best = pop[np.argmin(fitness)]
        
        for i in range(len(pop)):
            pop, fitness = mutualism_step(pop, fitness, best, obj_func, map_func, i)
            best = pop[np.argmin(fitness)]
            pop, fitness = commensalism_step(pop, fitness, best, obj_func, map_func, i)
            best = pop[np.argmin(fitness)]
            pop, fitness = parasitism_step(pop, fitness, obj_func, map_func, i, 
                                           lb=lb, ub=ub, is_portfolio=is_portfolio)
            
        best_curve.append(float(np.min(fitness)))
        
    idx = np.argmin(fitness)
    return float(fitness[idx]), pop[idx], best_curve
