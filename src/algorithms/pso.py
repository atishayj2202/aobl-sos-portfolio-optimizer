import numpy as np

def PSO(obj_func, pop: np.ndarray, map_func, iters: int, 
        lb: float = None, ub: float = None, is_portfolio: bool = False):
    """
    Standard Particle Swarm Optimization (PSO) with velocity clamping.
    """
    pop = pop.copy()
    pop_size, dim = pop.shape
    
    # Initialize velocities
    if is_portfolio:
        v_max = 0.1
    else:
        v_max = 0.1 * (ub - lb)
    velocities = np.random.uniform(-v_max, v_max, (pop_size, dim))
    
    # Initialize personal bests
    pbest = pop.copy()
    pbest_fit = np.array([obj_func(ind) for ind in pbest], dtype=float)
    
    gbest_idx = np.argmin(pbest_fit)
    gbest = pbest[gbest_idx].copy()
    gbest_fit = pbest_fit[gbest_idx]
    
    # Parameters (Clerc & Kennedy constriction coefficients)
    w = 0.7298
    c1 = 1.49618
    c2 = 1.49618
    
    best_curve = []
    
    for t in range(iters):
        for i in range(pop_size):
            r1 = np.random.uniform(0, 1, dim)
            r2 = np.random.uniform(0, 1, dim)
            
            # Velocity update
            velocities[i] = (w * velocities[i] + 
                             c1 * r1 * (pbest[i] - pop[i]) + 
                             c2 * r2 * (gbest - pop[i]))
            
            # Clamping velocity
            velocities[i] = np.clip(velocities[i], -v_max, v_max)
            
            # Position update
            pop[i] = map_func(pop[i] + velocities[i])
            
            # Evaluation
            fit = obj_func(pop[i])
            
            if fit < pbest_fit[i]:
                pbest[i] = pop[i].copy()
                pbest_fit[i] = fit
                
                if fit < gbest_fit:
                    gbest = pop[i].copy()
                    gbest_fit = fit
                    
        best_curve.append(float(gbest_fit))
        
    return float(gbest_fit), gbest, best_curve
