#%%
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from lib.pdf import PDF

class Classifier:
    def __init__(self, q_slot_multiplier=30, parameters=["rmag", "rabsmag", "redshift", "rad", "sern"]):
        self.q_slot_multiplier = q_slot_multiplier
        self.parameters = parameters

        self.clf = RandomForestClassifier(
            criterion="gini",
            n_estimators=64,
            #max_depth=10,
            min_samples_split=10,
            min_samples_leaf=25,
            max_features=None,
            n_jobs=-1
        )
    
    def fit(self, galaxies):
        q_slots = galaxies["ba"].multiply(self.q_slot_multiplier).apply(np.ceil).astype(int) - 1
        self.clf.fit(galaxies[self.parameters].values, q_slots.values)
    
    def predict_pdf(self, galaxy):
        pdf = self.clf.predict_proba(galaxy[self.parameters].values.reshape(1, -1))[0] * self.q_slot_multiplier
        return PDF(np.linspace(0, 1, self.q_slot_multiplier) + 1/self.q_slot_multiplier/2, pdf)
