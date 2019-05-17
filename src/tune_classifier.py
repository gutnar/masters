#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lib import Classifier, PDF, BayesianApproximation2d
from lib.plotting import *
from sklearn.model_selection import RandomizedSearchCV


def evaluate(estimator, X, y):
    return np.sum(estimator.predict_proba(X)[range(len(y)), y]) / len(y)


def test_parameter(param, values, kwargs={}):
    train_results = []
    test_results = []

    for value in values:
        kwargs[param] = value
        rf = Classifier(**kwargs)
        rf.fit(train_galaxies)
        train_results.append(rf.evaluate(train_galaxies))
        test_results.append(rf.evaluate(test_galaxies))
        print(value)
    
    plt.figure(1)
    plt.title("%s train score" % param)
    plt.plot(values, train_results)

    plt.figure(2)
    plt.title("%s test score" % param)
    plt.plot(values, test_results)
    plt.legend()

#%% Load galaxies
galaxies = pd.read_csv("data/raw/data_gama_gal_orient.txt", r"\s+")
spiral_galaxies = pd.read_csv("data/raw/gama_spiral.txt", r"\s+")
elliptic_galaxies = pd.read_csv("data/raw/gama_elliptic.txt", r"\s+")

test_galaxies = pd.merge(pd.concat((spiral_galaxies, elliptic_galaxies)), galaxies, on="id")
train_galaxies = galaxies[~galaxies["id"].isin(test_galaxies["id"])]

len(train_galaxies), len(test_galaxies)

#%%
p = []
s = (10, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100)
c = [0.5885791171856672] + list((1/np.array(s))[1:])

for slots in s:
    q_pdf = PDF.from_samples(np.linspace(0, 1, slots), train_galaxies["ba"])
    q_slots = test_galaxies["ba"].multiply(slots).apply(np.ceil).astype(int) - 1
    p.append(np.sum(q_pdf.y[q_slots]) / slots / len(q_slots))

plt.plot(s, 1/np.array(s))
plt.plot(s, p)
#plt.plot(s, c)

p[2]

#%%
galaxies = pd.read_csv("data/intermediate/galaxies.csv")
#np.random.seed(0b11011100000011011011000001110000)
training_set = np.random.rand(len(galaxies)) < 0.75
train_galaxies = galaxies[training_set]
test_galaxies = galaxies[~training_set]

len(train_galaxies), len(test_galaxies)

#%% Manual fit
classifier = Classifier(25, n_estimators=6, max_depth=10, max_features=2, bootstrap=True, criterion="entropy")
classifier.fit(train_galaxies)
classifier.evaluate(train_galaxies), classifier.evaluate(test_galaxies)

#%% Manual fit
p1 = []
p2 = []
s = (10, 11, 13, 14, 15, 17, 20, 22, 25, 27, 30, 35, 40, 50, 60, 70, 80, 100)

for slots in s:
    classifier = Classifier(slots, n_estimators=6, max_depth=10, max_features=2, bootstrap=True, criterion="entropy")
    classifier.fit(train_galaxies)
    p1.append(classifier.evaluate(test_galaxies))

    q_pdf = PDF.from_samples(np.linspace(0, 1, slots), train_galaxies["ba"])
    q_slots = test_galaxies["ba"].multiply(slots).apply(np.ceil).astype(int) - 1
    p2.append(np.sum(q_pdf.y[q_slots]) / slots / len(q_slots))

#%%
plt.plot(np.linspace(10, 100, 90), 1/np.array(np.linspace(10, 100, 90)), label="Ühtlase q jaotuse järgi")
plt.plot(s, p2, label="Treeningandmete q jaotuse järgi")
plt.plot(s, p1, label="Treenitud otsustusmetsa järgi")
plt.legend()

plt.savefig("plots/global_vs_classifier.pdf")

#%%
p1[8] - p2[8]

#%%
classifier.clf.feature_importances_

#%%
for i in (10, 80, 190):
    predicted_pdf = classifier.predict_pdf(test_galaxies.iloc[[i]])
    plt.plot(predicted_pdf.x, predicted_pdf.y)

plt.savefig("plots/random_predicted_pdf.pdf")

#%% Parameter distribution for random search
param_distributions = {
    "n_estimators": list(range(5, 25)),
    "max_depth": list(range(7, 25)),
    "min_samples_split": np.linspace(100, 1000, 10, dtype=int),
    "min_samples_leaf": np.linspace(100, 1000, 10, dtype=int),
    "max_features": [3, 4, 5],
    "bootstrap": [True, False]
}

#%%
rs_classifier = Classifier(25)
rs_classifier.clf = RandomizedSearchCV(
    estimator=classifier.clf,
    param_distributions=param_distributions,
    n_iter=1000,
    cv=5,
    verbose=2,
    n_jobs=-1,
    scoring=evaluate
)

rs_classifier.fit(train_galaxies)
rs_classifier.clf = rs_classifier.clf.best_estimator_

print(rs_classifier.clf)
print(rs_classifier.evaluate(train_galaxies), rs_classifier.evaluate(test_galaxies))

#%%
test_parameter("n_estimators", [8, 9, 10, 11, 12, 13, 14, 15, 16], {
    "max_depth": 11,
    "max_features": 4,
    "min_samples_leaf": 100,
    "min_samples_split": 600,
    "bootstrap": False
})

#%%
train_scores = []
test_scores = []

for m in (2, 4, 8, 16, 32, 64, 128):
    m_classifier = Classifier(20, 7, 30, m, 1, 5)
    m_classifier.fit(train_galaxies)

    train_scores.append(m_classifier.evaluate(train_galaxies))
    test_scores.append(m_classifier.evaluate(test_galaxies))

plt.figure(1)
plt.plot(train_scores)

plt.figure(2)
plt.plot(test_scores)
