import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
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

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 42, train_size = 0.80)

# Define the hyperparameter search space
param_space = {
    'n_estimators': (50, 200),  # Number of trees in the forest
    'max_depth': (10, 300)        # Maximum depth of the trees
}   

# Define the objective function
def objective_function(params): 
     # Convert max_depth to integer
    params['max_depth'] = int(params['max_depth'])

    params['n_estimators'] = int(params['n_estimators'])
    # Create a Random Forest classifier with the given hyperparameters
    clf = RandomForestClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        random_state=42
    )
    
    # Evaluate the model using cross-validation (you can use your own dataset)
    scores = cross_val_score(clf, X_train, Y_train, cv=10, scoring='accuracy')
    
    # Return the mean accuracy as the objective value
    return np.mean(scores)

# Cuckoo Search parameters
population_size = 10    
max_iterations = 20
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
            nests[i] = {param: np.random.uniform(param_space[param][0], param_space[param][1]) for param in param_space}

# Find the best nest (best hyperparameters)
best_nest = max(nests, key=objective_function)
best_accuracy = objective_function(best_nest)

print("Best Hyperparameters:", best_nest)
print("Best Accuracy:", best_accuracy)

#Best Hyperparameters: {'n_estimators': 108, 'max_depth': 10}
#Best Accuracy: 0.6387141858839972