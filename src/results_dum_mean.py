#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
galaxies = pd.read_csv("data/final/filament_galaxies_random_dum_mean.csv")
spiral = galaxies[galaxies["dum_mean"] <= 0.5]
elliptic = galaxies[galaxies["dum_mean"] > 0.5]

#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from subprocess import call

classifier = RandomForestClassifier(
    n_estimators=1,
    max_depth=2,
    #min_samples_split=10,
    #min_samples_leaf=10,
    max_features=2,
    n_jobs=-1
)

columns = ["rmag", "rabsmag", "redshift", "rad", "sern", "ba"]
X = galaxies[columns].values
y = np.select((
    galaxies["dum_mean"] <= 0.5,
    galaxies["dum_mean"] > 0.5
), (0, 1))
sample_weight = np.abs(galaxies["dum_mean"] - 0.5)

classifier.fit(X, y, sample_weight)
classifier.score(X, y)

export_graphviz(
    classifier.estimators_[0],
    feature_names=columns,
    class_names=("spiral", "elliptic"),
    filled=True, rounded=True,
    out_file="plots/tree.dot"
)

call(['graphviz/dot', '-Tpng', 'plots/tree.dot', '-o', 'plots/tree.png', '-Gdpi=600'])
