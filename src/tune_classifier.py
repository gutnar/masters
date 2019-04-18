#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lib import Classifier, PDF, BayesianApproximation
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

#%% Manual fit
classifier = Classifier()
classifier.fit(train_galaxies)
classifier.evaluate(train_galaxies), classifier.evaluate(test_galaxies)

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
plt.plot(rs_classifier.predict_pdf(test_galaxies.iloc[[120]]).y)

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

#%%
ba = BayesianApproximation(rs_classifier.predict_pdf(test_galaxies.iloc[[1290]]))
ba.run([(150000, 0.05)]*25)

#%%
plot_ba_results(ba)

#%%
plot_xz_kde(ba)

#%%
plot_qt_kde(ba)

#%%
plot_qp_kde(ba)

#%%
t_pdf = ba.get_t_pdf(0.8)
plt.plot(t_pdf.x, t_pdf.y, label=r"$\theta$")
plt.legend()

#%%
p_pdf = ba.get_p_pdf(0.8)
plt.plot(p_pdf.x, p_pdf.y, label=r"$\phi$")
plt.hist(p_pdf.sample(10000), 100, (-np.pi/2, np.pi/2), True)
plt.legend()
