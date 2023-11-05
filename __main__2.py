import numpy as np
from scipy.special import gamma

def cuckoo_search(inp=None):
    if inp is None:
        inp = [25, 1000]  # Default values for n and N_IterTotal
    
    n = inp[0]  # Population size
    N_IterTotal = inp[1]  # Change this if you want to get better results
    pa = 0.25  # Discovery rate of alien eggs/solutions
    nd = 15  # Dimensions of the problem
    
    # Simple bounds of the search domain
    Lb = -5 * np.ones(nd)  # Lower bounds
    Ub = 5 * np.ones(nd)   # Upper bounds
    
    # Random initial solutions
    nest = [Lb + (Ub - Lb) * np.random.rand(nd) for _ in range(n)]
    nest = np.array(nest)
    
    # Get the current best of the initial population
    fitness = 1e10 * np.ones(n)
    fmin, bestnest, nest, fitness = get_best_nest(nest, nest, fitness)
    
    # Starting iterations
    for iter in range(N_IterTotal):
        # Generate new solutions (but keep the current best)
        new_nest = get_cuckoos(nest, bestnest, Lb, Ub)
        fnew, best, nest, fitness = get_best_nest(nest, new_nest, fitness)
        
        # Discovery and randomization
        new_nest = empty_nests(nest, Lb, Ub, pa)
        
        # Evaluate this set of solutions
        fnew, best, nest, fitness = get_best_nest(nest, new_nest, fitness)
        
        # Find the best objective so far
        if fnew < fmin:
            fmin = fnew
            bestnest = best
        
        # Display the results every 100 iterations
        if iter % 100 == 0:
            print(f'Iteration = {iter}')
            print(f'Bestnest = {bestnest}')
            print(f'Fmin = {fmin}')
    
    # Post-optimization processing and display all the nests
    print(f'The best solution: {bestnest}')
    print(f'The best fmin: {fmin}')

def get_cuckoos(nest, best, Lb, Ub):
    n = len(nest)
    beta = 1.5
    sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta) 
    
    for j in range(n):
        s = nest[j]
        u = np.random.randn(*s.shape) * sigma
        v = np.random.randn(*s.shape)
        step = u / abs(v)**(1 / beta)
        stepsize = 0.01 * step * (s - best)
        s = s + stepsize * np.random.randn(*s.shape)
        nest[j] = simplebounds(s, Lb, Ub)
    
    return nest

def get_best_nest(nest, newnest, fitness):
    for j in range(len(nest)):
        fnew = fobj(newnest[j])
        if fnew <= fitness[j]:
            fitness[j] = fnew
            nest[j] = newnest[j]
    
    fmin = np.min(fitness)
    best_index = np.argmin(fitness)
    best = nest[best_index]

    return fmin, best, nest, fitness

def empty_nests(nest, Lb, Ub, pa):
    n = len(nest)
    K = np.random.rand(n) > pa
    stepsize = np.random.rand(*nest.shape) * (nest[np.random.permutation(n)] - nest[np.random.permutation(n)])
    new_nest = nest + stepsize * K[:, np.newaxis]
    
    for j in range(len(new_nest)):
        new_nest[j] = simplebounds(new_nest[j], Lb, Ub)
    
    return new_nest

def simplebounds(s, Lb, Ub):
    s = np.maximum(s, Lb)
    s = np.minimum(s, Ub)
    return s

def fobj(u):
    return np.sum((u - 1)**2)

if __name__ == "__main__":
    cuckoo_search()
