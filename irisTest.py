import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.model_selection import cross_val_score
import pandas as pd
from numba import jit, cuda


# Load the Iris dataset (as an example)
dataset = pd.read_excel('Denguedatasample1.xlsx')


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
y = dataset.FINAL

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the hyperparameter search space for Random Forest
hyperparameter_space = {
    'n_estimators': [50, 100, 150,200],
    'max_depth': [50,60,70,80,90,100,110,120,130,140,150],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'random_state': [42]
}

# Define the objective function (accuracy of Random Forest)
def objective_function(hyperparameters):
    model = RandomForestClassifier(**hyperparameters)
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_val)
    accuracy = accuracy_score(y_val,y_pred_test)
    #accuracy = model.score(X_val, y_val)
    
    return accuracy


#print(objective_function({'n_estimators': 100, 'max_depth': 38, 'min_samples_split': 2, 'min_samples_leaf': 3, 'random_state': 42}))
#model2=RandomForestClassifier(criterion='entropy', bootstrap=True, max_depth=100, random_state=0)
#Best Hyperparameters: {'n_estimators': 50, 'max_depth': 27, 'min_samples_split': 10, 'min_samples_leaf': 1, 'random_state': 42}
#model2.fit(X_train,y_train)
#y_pred_test = model2.predict(X_val)
#print(accuracy_score(y_val, y_pred_test))
#scores = cross_val_score(model2, X, y, cv=10, scoring='accuracy')
#print(scores.mean())

#model3= RandomForestClassifier(criterion='entropy', bootstrap=True, max_depth=100, random_state=0)
#model3.fit(X_train,y_train)
#y_pred_test = model3.predict(X_val)
#print(accuracy_score(y_val, y_pred_test))

# Cuckoo Search hyperparameter tuning
def cuckoo_search(objective_function, hyperparameter_space, population_size=10, max_iterations=500):
    population = []

    # Initialize the population with random hyperparameters
    for _ in range(population_size):
        hyperparameters = {}
        for param, values in hyperparameter_space.items():
            hyperparameters[param] = np.random.choice(values)
        population.append((hyperparameters, objective_function(hyperparameters)))

    best_solution = max(population, key=lambda x: x[1])

    for iteration in range(max_iterations):
        new_population = []

        for cuckoo in population:
            current_hyperparameters, current_fitness = cuckoo

            # Perform a random walk to generate a new solution
            step_size = 1.0
            new_hyperparameters = {}
            for param, value in current_hyperparameters.items():
                new_value = value + step_size * np.random.randn()
                new_value = np.clip(new_value, min(hyperparameter_space[param]), max(hyperparameter_space[param]))
                new_value = int(new_value)  # Round to the nearest integer
                new_hyperparameters[param] = new_value

            # Evaluate the new solution
            new_fitness = objective_function(new_hyperparameters)

            # Replace the cuckoo if the new solution is better
            if new_fitness > current_fitness:
                new_population.append((new_hyperparameters, new_fitness))
            else:
                new_population.append(cuckoo)

            # Update the best solution
            if new_fitness > best_solution[1]:
                best_solution = (new_hyperparameters, new_fitness)

        population = new_population

        print(f"Iteration {iteration+1}/{max_iterations}: Best Accuracy = {objective_function(new_hyperparameters)} Parameters: {new_hyperparameters}")

    return best_solution

# Hyperparameter tuning using Cuckoo Search
best_hyperparameters, best_accuracy = cuckoo_search(objective_function, hyperparameter_space, population_size=10, max_iterations=500)

print("Best Hyperparameters:", best_hyperparameters)
print("Best Accuracy:", best_accuracy)