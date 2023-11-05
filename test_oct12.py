import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from joblib import Parallel, delayed

import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_excel('Denguedatasample1.xlsx')

#print dataset to check if it reads the correct dataset
#comment this line if you are getting the correct dataset
#print(dataset)

target = []
#initializing the features
features = []
for i in range(len(dataset.columns)):
    if i < len(dataset.columns)-1:
        features.append(dataset.columns[i])
    else:
        target.append(dataset.columns[i])

#print(features)
#print(target)
X = dataset[features] 

Y = dataset.FINAL

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 0, train_size = 0.80)

# Define the hyperparameter search space (similar to Cuckoo Search)
param_space = {
    'n_estimators': (10, 100),      # Number of trees in the forest
    'max_depth': (5, 50),           # Maximum depth of the trees
    'min_samples_split': (2, 11),   # Minimum number of samples required to split an internal node
    'min_samples_leaf': (1, 11),    # Minimum number of samples required to be at a leaf node
    'max_features': (1, 64),    # Fraction of features to consider for split
}



# Grid Search
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=0),
    param_grid=param_space,
    cv=10,
    scoring='accuracy',
    n_jobs=-1  # Use all available CPU cores for parallelization
)
grid_search.fit(X_train, Y_train)

#Random Search
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=0),
    param_distributions=param_space,
    n_iter=50,  # Number of random parameter combinations to try
    cv=10,
    scoring='accuracy',
    random_state=0
)
random_search.fit(X_train, Y_train)

# Display the results
print("Grid Search Best Hyperparameters:", grid_search.best_params_)
print("Grid Search Best Accuracy:", grid_search.best_score_)
print("Random Search Best Hyperparameters:", random_search.best_params_)
print("Random Search Best Accuracy:", random_search.best_score_)



# Define the objective function
def objective_function(params): 
     # Convert max_depth to integer
    params['max_depth'] = int(params['max_depth'])
    params['min_samples_split'] = int(params['min_samples_split'])
    params['min_samples_leaf'] = int(params['min_samples_leaf'])
    params['n_estimators'] = int(params['n_estimators'])
    # Create a Random Forest classifier with the given hyperparameters
    clf = RandomForestClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split = params['min_samples_split'],
        min_samples_leaf = params['min_samples_leaf'],
        random_state=42 
    )
    # Evaluate the model using cross-validation (you can use your own dataset)
    scores = cross_val_score(clf, X_train, Y_train, cv=10, scoring='accuracy')
    
    # Return the mean accuracy as the objective value
    return np.mean(scores)


def levy_flight(current_value, param):
    # Define the scale factor for Levy Flight (you can adjust it based on your problem)
    scale = 0.1

    # Generate a random step from the Levy distribution
    # The parameter 1.5 is used for the Levy exponent (can be adjusted)
    step = scale * np.random.normal(0, (1.5 / abs(np.random.normal())) ** (1/1.5))

    # Update the current value with the Levy step
    new_value = current_value + step

    # Ensure the new value is within the parameter space
    new_value = max(param_space[param][0], min(param_space[param][1], new_value))

    return new_value

# Cuckoo Search parameters
population_size = 10    
max_iterations = 50
pa = 0.25  # Probability of abandoning a nest
alpha = 1  # Step size scaling factor

# Initialize cuckoo nests randomly within the search space
nests = []
for _ in range(population_size):
    nest = {param: np.random.uniform(param_space[param][0], param_space[param][1]) for param in param_space} 
    nests.append(nest)

# Optimization loop
for iteration in range(max_iterations):
    for i in range(population_size):
        # Generate a new solution using Levy Flight
        new_nest = {param: levy_flight(nest[param], param) for param in nest}
        # Evaluate the new nest's objective value
        new_nest_value = objective_function(new_nest)

        # Select a random nest to replace
        j = np.random.randint(0, population_size)

        # Replace the nest with the new nest if it has a better objective value
        if new_nest_value > objective_function(nests[j]):
            nests[j] = new_nest

    # Randomly choose a fraction of nests to abandon and replace them with new random nests
    for i in range(population_size):
        if np.random.rand() < pa:
            new_nest = {}
            for param in param_space:
                # Ensure the low value is less than the high value for each parameter
                low, high = param_space[param]

                # Check if low is greater than or equal to high, and swap them if necessary
                if low >= high:
                    low, high = high, low + 1  # Adding 1 to high to ensure it's greater than low

                new_nest[param] = np.random.randint(low, high)
            nests[i] = new_nest


# Find the best nest (best hyperparameters)
best_nest = max(nests, key=objective_function)
best_accuracy = objective_function(best_nest)

print("Best Hyperparameters:", best_nest)
print("Best Accuracy:", best_accuracy)

#Best Hyperparameters: {'n_estimators': 108, 'max_depth': 10}
#Best Accuracy: 0.6387141858839972


#Best Hyperparameters: {'n_estimators': 438, 'max_depth': 40, 'min_samples_split': 18, 'min_samples_leaf': 7, 'max_features': 0, 'bootstrap': 0}:
#Best Accuracy: 0.6461215932914046