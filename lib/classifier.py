#%%
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from lib.pdf import PDF

#%%
class Classifier:
    def __init__(self,
        q_slot_multiplier=25,
        n_estimators=11,
        max_depth=11,
        min_samples_split=600,
        min_samples_leaf=100,
        max_features=4,
        bootstrap=False,
        parameters=["rmag", "rabsmag", "redshift", "rad", "sern"]):
        self.q_slot_multiplier = q_slot_multiplier
        self.parameters = parameters

        self.clf = RandomForestClassifier(
            criterion="gini",
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            n_jobs=-1
        )
    
    def fit(self, galaxies):
        q_slots = galaxies["ba"].multiply(self.q_slot_multiplier).apply(np.ceil).astype(int) - 1
        self.clf.fit(galaxies[self.parameters].values, q_slots.values)
    
    def score(self, galaxies):
        q_slots = galaxies["ba"].multiply(self.q_slot_multiplier).apply(np.ceil).astype(int) - 1
        return self.clf.score(galaxies[self.parameters].values, q_slots.values)
    
    def evaluate(self, galaxies):
        q_slots = galaxies["ba"].multiply(self.q_slot_multiplier).apply(np.ceil).astype(int) - 1
        return np.sum(
            self.clf.predict_proba(galaxies[self.parameters].values)[range(len(q_slots)), q_slots]
        ) / len(galaxies)
    
    def predict_pdf(self, galaxy):
        pdf = self.clf.predict_proba(galaxy[self.parameters].values.reshape(1, -1))[0] * self.q_slot_multiplier
        return PDF(np.linspace(0, 1, self.q_slot_multiplier) + 1/self.q_slot_multiplier/2, pdf)

#%%
import matplotlib.pyplot as plt

def test_parameter(train_galaxies, test_galaxies, param, values):
    kwargs = {}
    train_results = []
    test_results = []

    for value in values:
        kwargs[param] = value
        rf = Classifier(**kwargs)
        rf.fit(train_galaxies)
        train_results.append(rf.score(train_galaxies))
        test_results.append(rf.score(test_galaxies))
        print(value)
    
    plt.figure(1)
    plt.title("%s train score" % param)
    plt.plot(values, train_results)

    plt.figure(2)
    plt.title("%s test score" % param)
    plt.plot(values, test_results)
    plt.legend()

#%%
if __name__ == "__main__":
    test_galaxies = pd.read_csv("data/intermediate/test_galaxies.csv")
    train_galaxies = pd.read_csv("data/intermediate/train_galaxies.csv")

#%% n_estimators = 11 #16
if __name__ == "__main__":
    test_parameter(
        train_galaxies, test_galaxies,
        "n_estimators", list(range(1, 17))
    )

#%% max_depth = 10
if __name__ == "__main__":
    test_parameter(
        train_galaxies, test_galaxies,
        "max_depth", list(range(1, 33))
    )

#%% min_samples_split = 0.01
if __name__ == "__main__":
    test_parameter(
        train_galaxies, test_galaxies,
        "min_samples_split", np.linspace(0.001, 0.1, 10, endpoint=True)
    )

#%% min_samples_leaf = 0.025
if __name__ == "__main__":
    test_parameter(
        train_galaxies, test_galaxies,
        "min_samples_leaf", np.linspace(0.001, 0.1, 10, endpoint=True)
    )

#%% max_features = 3
if __name__ == "__main__":
    test_parameter(
        train_galaxies, test_galaxies,
        "max_features", list(range(1, 6))
    )
