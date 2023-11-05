import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from joblib import Parallel, delayed

dataset = pd.read_excel('Denguedatasample1.xlsx')

target = []
features = []
for i in range(len(dataset.columns)):
    if i < len(dataset.columns) - 1:
        features.append(dataset.columns[i])
    else:
        target.append(dataset.columns[i])

X = dataset[features]
Y = dataset.FINAL

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0, train_size=0.80)

param_space = {
    'n_estimators': (50, 500),
    'max_depth': (1, 50),
    'min_samples_split': (2, 20),
    'min_samples_leaf': (1, 20),
    'max_features': (0.1, 1.0),
    'bootstrap': (True, False)
}

def objective_function(params):
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
    
    scores = cross_val_score(clf, X_train, Y_train, cv=10, scoring='neg_log_loss')
    
    return np.mean(scores)

def levy_flight(current_value, param):
    scale = 0.1
    step = scale * np.random.normal(0, (1.5 / abs(np.random.normal())) ** (1/1.5))
    new_value = current_value + step
    new_value = max(param_space[param][0], min(param_space[param][1], new_value))
    return new_value

def evaluate_nest(nest):
    return nest, objective_function(nest)

population_size = 10
max_iterations = 50
pa = 0.25
alpha = 1

nests = []
for _ in range(population_size):
    nest = {param: np.random.uniform(param_space[param][0], param_space[param][1]) for param in param_space}
    nests.append(nest)

if __name__ == '__main__':
    for iteration in range(max_iterations):
        nest_results = Parallel(n_jobs=-1)(
            delayed(evaluate_nest)(nest) for nest in nests
        )

        for i, (nest, value) in enumerate(nest_results):
            if value > objective_function(nests[i]):
                nests[i] = nest

    best_nest, best_value = min(nest_results, key=lambda x: x[1])

    print("Best Hyperparameters:", best_nest)
    print("Best Accuracy:", best_value)
    
    # Create a dictionary to specify the parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [50, 500],
        'max_depth': list(range(1, 51)),
        'min_samples_split': list(range(2, 21)),
        'min_samples_leaf': list(range(1, 21)),
        'max_features': np.arange(0.1, 1.1, 0.1),
        'bootstrap': [True, False]
    }

    clf = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=10, scoring='neg_log_loss', n_jobs=-1)
    grid_search.fit(X_train, Y_train)

    best_grid_search_loss = -grid_search.best_score_  # Since GridSearchCV returns the negative log loss

    # Compare the results
    if best_cuckoo_loss < best_grid_search_loss:
        print("Cuckoo Search found a better set of hyperparameters.")
        print("Best Hyperparameters (Cuckoo Search):", best_cuckoo_nest)
        print("Best Log Loss (Cuckoo Search):", best_cuckoo_loss)
    else:
        print("GridSearchCV found a better set of hyperparameters.")
        print("Best Hyperparameters (GridSearchCV):", grid_search.best_params_)
        print("Best Log Loss (GridSearchCV):", best_grid_search_loss)