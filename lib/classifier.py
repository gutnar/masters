#%%
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from lib.pdf import PDF

#%%
class Classifier:
    def __init__(self,
        q_slot_multiplier=30,
        n_estimators=100,#11,
        max_depth=6,#11,
        min_samples_split=600,
        min_samples_leaf=100,
        max_features=2,#4,
        bootstrap=True,#False,
        parameters=["rmag", "rabsmag", "redshift", "rad", "sern"],
        criterion="entropy"#"gini"
    ):
        self.q_slot_multiplier = q_slot_multiplier
        self.parameters = parameters

        self.clf = RandomForestClassifier(
            criterion=criterion,
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
        return PDF(np.linspace(0, 1, self.q_slot_multiplier, endpoint=False) + 1/self.q_slot_multiplier/2, pdf)
