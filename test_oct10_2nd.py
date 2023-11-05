import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV

import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_excel('Denguedatasample1.xlsx')

#print dataset to check if it reads the correct dataset
#comment this line if you are getting the correct dataset
#print(dataset)
#separating the features and output values (target)
#initializing the target
target = []
#initializing the features
features = []
for i in range(len(dataset.columns)):
    if i < len(dataset.columns)-1:
        features.append(dataset.columns[i])
    else:
        target.append(dataset.columns[i])


#checking if you have separated them correctly
#comment the next two lines if you are done checking
#print(features)
#print(target)
X = dataset[features] 

Y = dataset.FINAL

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 0, train_size = 0.80)

# Define the hyperparameter search space (similar to Cuckoo Search)
param_space = {
    'n_estimators': (50, 500),      # Number of trees in the forest
    'max_depth': (1, 50),           # Maximum depth of the trees
    'min_samples_split': (2, 20),   # Minimum number of samples required to split an internal node
    'min_samples_leaf': (1, 20),    # Minimum number of samples required to be at a leaf node
    'max_features': (0.1, 1.0),    # Fraction of features to consider for split
    'bootstrap': (True, False)      # Whether to bootstrap samples
}



# Grid Search
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=0),
    param_grid=param_space,
    cv=5,
    scoring='log_loss',
    n_jobs=-1  # Use all available CPU cores for parallelization
)
grid_search.fit(X_train, Y_train)

# Random Search
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=0),
    param_distributions=param_space,
    n_iter=50,  # Number of random parameter combinations to try
    cv=5,
    scoring='log_loss',
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
        # Generate a random cuckoo egg by perturbing the nest's hyperparameters
        egg = {param: nest[param] + alpha * np.random.randn() for param in nest}
        
        # Ensure the egg's hyperparameters are within the search space
        for param in param_space:
            egg[param] = max(param_space[param][0], min(param_space[param][1], egg[param]))
        
        # Evaluate the egg's objective value
        egg_value = objective_function(egg)
        
        # Select a random nest to replace
        j = np.random.randint(0, population_size)

        # Replace the nest with the egg if the egg's objective value is better
        if egg_value > objective_function(nests[j]):
            nests[j] = egg
    
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