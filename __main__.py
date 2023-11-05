import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
import multiprocessing  # Import multiprocessing module
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
    cv=10,
    scoring='accuracy',
    n_jobs=-1  # Use all available CPU cores for parallelization
)
grid_search.fit(X_train, Y_train)

# Random Search
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

# Cuckoo Search parameters
population_size = 10    
max_iterations = 50
pa = 0.25  # Probability of abandoning a nest
alpha = 0.1  # Step size scaling factor

# Initialize cuckoo nests randomly within the search space
nests = []
for _ in range(population_size):
    nest = {param: np.random.uniform(param_space[param][0], param_space[param][1]) for param in param_space} 
    nests.append(nest)

if __name__ == '__main__':  # Add this block to prevent multiprocessing errors
    # Call the optimization function from the cuckoo_search module with parallelization
    cuckoo_search.optimize_nests(nests, X_train, Y_train)

#Best Hyperparameters: {'n_estimators': 108, 'max_depth': 10}
#Best Accuracy: 0.6387141858839972