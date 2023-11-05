import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from hyperopt import fmin, tpe, hp
from sklearn.metrics import accuracy_score
import random

# Create a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Random Forest classifier with default hyperparameters
rf_classifier = RandomForestClassifier(random_state=42)

# Define the hyperparameter search space for Grid Search
grid_search_space = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 20, 30],
}

# Perform Grid Search
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=grid_search_space, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_grid_params = grid_search.best_params_
best_grid_accuracy = grid_search.best_score_

# Define the hyperparameter search space for Random Search (using hyperopt)
random_search_space = {
    'n_estimators': hp.choice('n_estimators', [50, 100, 150]),
    'max_depth': hp.choice('max_depth', [10, 20, 30]),
}

# Define the objective function for Random Search
# Define the hyperparameter search space for Random Search (using hyperopt)
random_search_space = {
    'n_estimators': hp.choice('n_estimators', [50, 100, 150]),
    'max_depth': hp.choice('max_depth', [10, 20, 30]),
}

# Define the objective function for Random Search
def random_search_objective(params):
    params['max_depth'] = int(params['max_depth'])  # Convert max_depth to an integer
    rf_classifier.set_params(**params)
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)
    return -accuracy_score(y_test, y_pred)  # Minimize negative accuracy


# Perform Random Search

random_search = fmin(fn=random_search_objective, space=random_search_space, algo=tpe.suggest, max_evals=30, rstate=random.seed(42))

best_random_params = {
    'n_estimators': [50, 100, 150][random_search['n_estimators']],
    'max_depth': [10, 20, 30][random_search['max_depth']]
}
rf_classifier.set_params(**best_random_params)  # Set the best hyperparameters
rf_classifier.fit(X_train, y_train)  # Fit the model with the best hyperparameters
y_pred = rf_classifier.predict(X_test)  # Predict with the best model
best_random_accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy

# Define the hyperparameter search space for Cuckoo Search
cuckoo_search_space = {
    'n_estimators': (50, 200),  # Number of trees in the forest
    'max_depth': (10, 100),    # Maximum depth of the trees
}

# Define the objective function for Cuckoo Search
def cuckoo_search_objective(params):
    # Ensure 'max_depth' is an integer within the valid range
    params['max_depth'] = int(params['max_depth'])
    params['n_estimators'] = int(params['n_estimators'])

    rf_classifier.set_params(**params)
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)
    return -accuracy_score(y_test, y_pred)  # Minimize negative accuracy


# Cuckoo Search parameters
cuckoo_population_size = 20
cuckoo_max_iterations = 50
cuckoo_pa = 0.25  # Probability of abandoning a nest
cuckoo_alpha = 1  # Step size scaling factor

# Initialize cuckoo nests randomly within the search space
cuckoo_nests = []
for _ in range(cuckoo_population_size):
    nest = {
        'n_estimators': np.random.randint(cuckoo_search_space['n_estimators'][0], cuckoo_search_space['n_estimators'][1] + 1),
        'max_depth': np.random.randint(cuckoo_search_space['max_depth'][0], cuckoo_search_space['max_depth'][1] + 1)
    }
    cuckoo_nests.append(nest)

# Optimization loop for Cuckoo Search
for iteration in range(cuckoo_max_iterations):
    for i in range(cuckoo_population_size):
        # Generate a random cuckoo egg by perturbing the nest's hyperparameters
        egg = {param: cuckoo_nests[i][param] + cuckoo_alpha * np.random.randn() for param in cuckoo_nests[i]}
        
        # Ensure the egg's hyperparameters are within the search space
        for param in cuckoo_search_space:
            egg[param] = max(cuckoo_search_space[param][0], min(cuckoo_search_space[param][1], egg[param]))
        
        # Evaluate the egg's objective value
        egg_value = cuckoo_search_objective(egg)
        
        # Select a random nest to replace
        j = np.random.randint(0, cuckoo_population_size)
        
        # Replace the nest with the egg if the egg's objective value is better
        if egg_value > cuckoo_search_objective(cuckoo_nests[j]):
            cuckoo_nests[j] = egg
    
    # Randomly choose a fraction of nests to abandon and replace them with new random nests
    for i in range(cuckoo_population_size):
        if random.random() < cuckoo_pa:
            nest = {param: np.random.uniform(cuckoo_search_space[param][0], cuckoo_search_space[param][1]) for param in cuckoo_search_space}
            cuckoo_nests[i] = nest

# Find the best nest (best hyperparameters)
best_cuckoo_params = max(cuckoo_nests, key=cuckoo_search_objective)
rf_classifier.set_params(**best_cuckoo_params)  # Set the best hyperparameters
rf_classifier.fit(X_train, y_train)  # Fit the model with the best hyperparameters
y_pred = rf_classifier.predict(X_test)  # Predict with the best model
best_cuckoo_accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy

# Print results
print("Best Hyperparameters (Grid Search):", best_grid_params)
print("Best Accuracy (Grid Search):", best_grid_accuracy)

print("Best Hyperparameters (Random Search):", best_random_params)
print("Best Accuracy (Random Search):", best_random_accuracy)

print("Best Hyperparameters (Cuckoo Search):", best_cuckoo_params)
print("Best Accuracy (Cuckoo Search):", best_cuckoo_accuracy)
