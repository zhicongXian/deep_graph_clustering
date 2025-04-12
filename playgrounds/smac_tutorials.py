# Created by zhicong.xian at 13:51 06.03.2025 using PyCharm

from ConfigSpace import Configuration, ConfigurationSpace

import numpy as np
from smac import HyperparameterOptimizationFacade, Scenario
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from ConfigSpace import Float, Categorical
iris = datasets.load_iris()


def train(config: Configuration, seed: int = 0) -> float:
    classifier = SVC(C=config["C"], random_state=seed)
    scores = cross_val_score(classifier, iris.data, iris.target, cv=5)
    return 1 - np.mean(scores)


configspace = ConfigurationSpace()#{"C": (0.100, 1000.0)})
c_config = Float("C", (0.0100, 1000.0), default=0.5)
kernel_config = Categorical("kernel", ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'])
configspace.add([c_config, kernel_config] )
# Scenario object specifying the optimization environment
scenario = Scenario(configspace, deterministic=True, n_trials=200)

# Use SMAC to find the best configuration/hyperparameters
smac = HyperparameterOptimizationFacade(scenario, train)
incumbent = smac.optimize()
print("debugging", incumbent)
best_score = smac.runhistory.get_min_cost(incumbent)
# best_parameters = incumbent.get_dictionary()
print(f"best_score found {1- best_score}")
import json

with open('data.json', 'w') as fp:
    json.dump(dict(incumbent), fp)