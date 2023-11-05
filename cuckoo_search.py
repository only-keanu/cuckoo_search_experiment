import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import multiprocessing

# Define the objective function for parallel evaluation
def parallel_objective_function(params, X_train, Y_train):
    params['max_depth'] = int(params['max_depth'])
    params['min_samples_split'] = int(params['min_samples_split'])
    params['min_samples_leaf'] = int(params['min_samples_leaf'])
    params['n_estimators'] = int(params['n_estimators'])
    
    clf = RandomForestClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        random_state=42
    )
    
    scores = cross_val_score(clf, X_train, Y_train, cv=10, scoring='accuracy')
    
    return np.mean(scores)

# ... (previous code) ...

# Optimization loop with parallelization
def optimize_nests(nests, X_train, Y_train):
    for iteration in range(max_iterations):
        pool = multiprocessing.Pool()  # Create a pool of worker processes
        results = []

        for i in range(population_size):
            egg = {param: nest[param] + alpha * np.random.randn() for param in nest}
            
            for param in param_space:
                egg[param] = max(param_space[param][0], min(param_space[param][1], egg[param]))
            
            # Evaluate the egg's objective value in parallel
            results.append(pool.apply_async(parallel_objective_function, (egg, X_train, Y_train)))

        # Close the pool and wait for all processes to finish
        pool.close()
        pool.join()

        for i in range(population_size):
            egg_value = results[i].get()
            j = np.random.randint(0, population_size)
            
            if egg_value > parallel_objective_function(nests[j], X_train, Y_train):
                nests[j] = egg

        for i in range(population_size):
            if np.random.rand() < pa:
                new_nest = {}
                for param in param_space:
                    low, high = param_space[param]
                    if low >= high:
                        low, high = high, low + 1
                    new_nest[param] = np.random.randint(low, high)
                nests[i] = new_nest

    best_nest = max(nests, key=lambda nest: parallel_objective_function(nest, X_train, Y_train))
    best_accuracy = parallel_objective_function(best_nest, X_train, Y_train)

    print("Best Hyperparameters:", best_nest)
    print("Best Accuracy:", best_accuracy)

if __name__ == '__main__':  # Add this block to prevent multiprocessing errors
    # Call the optimization function with parallelization
    optimize_nests(nests, X_train, Y_train)
