import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV

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
            new_nest = nest.copy()
            for param in param_space:
                new_nest[param] = levy_flight(nest[param], param)
            new_value = objective_function(new_nest)

            if new_value > value:
                nests[i] = new_nest

    best_nest, best_value = min(nest_results, key=lambda x: x[1])

    print("Best Hyperparameters Cuckoo:", best_nest)
    print("Best Accuracy Cuckoo:", best_value)

    clf_cuckoo = RandomForestClassifier(**best_nest, random_state=42)
    cv_scores_cuckoo = cross_val_score(clf_cuckoo, X_train, Y_train, cv=10, scoring='neg_log_loss')
    std_dev_cuckoo = np.std(cv_scores_cuckoo)
    print("Standard Deviation of Neg-log-loss per Fold (Cuckoo Search):", std_dev_cuckoo)

    param_grid = {
        'n_estimators': (50, 500),
        'max_depth': (1, 50),
        'min_samples_split': (2, 20),
        'min_samples_leaf': (1, 20),
        'max_features': (0.1, 1.0),
    }

    clf_grid = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(estimator=clf_grid, param_grid=param_grid, cv=10, scoring='neg_log_loss', n_jobs=-1)
    grid_search.fit(X_train, Y_train)

    cv_scores_grid = cross_val_score(clf_grid, X_train, Y_train, cv=10, scoring='neg_log_loss')
    std_dev_grid = np.std(cv_scores_grid)
    print("Standard Deviation of Neg-log-loss per Fold (GridSearchCV):", std_dev_grid)

    best_grid_roc = grid_search.best_score_  # Since GridSearchCV returns the negative log loss
    print("Best Hyperparameters Grid:", grid_search.best_params_)
    print("Best ROC_AUC Grid:", best_grid_roc)

    if best_value < best_grid_roc:
        print("Cuckoo Search found a better set of hyperparameters.")
        print("Best Hyperparameters (Cuckoo Search):", best_nest)
        print("Best Log Loss (Cuckoo Search):", best_value)
    else:
        print("GridSearchCV found a better set of hyperparameters.")
        print("Best Hyperparameters (GridSearchCV):", grid_search.best_params_)
        print("Best Log Loss (GridSearchCV):", best_grid_roc)
