#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV

from lib import Classifier, PDF, BayesianApproximation2d
from lib.plotting import *
from src.tex_plot import savefig


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
galaxies = pd.read_csv("data/intermediate/galaxies.csv")
np.random.seed(0b11011101100011011011011001110010)
training_set = np.random.rand(len(galaxies)) < 0.75
train_galaxies = galaxies[training_set]
test_galaxies = galaxies[~training_set]

train_galaxies.to_csv("data/intermediate/train_galaxies.csv", index=False)
test_galaxies.to_csv("data/intermediate/test_galaxies.csv", index=False)

len(train_galaxies), len(test_galaxies)

#%%
galaxies = pd.read_csv("data/intermediate/galaxies.csv")
train_galaxies = pd.read_csv("data/intermediate/train_galaxies.csv")
test_galaxies = pd.read_csv("data/intermediate/test_galaxies.csv")

len(train_galaxies), len(test_galaxies)

#%%
filament_galaxies = pd.read_csv("data/intermediate/filament_galaxies.csv")
filament_galaxies.describe()

#%% Manual fit
classifier = Classifier(30, n_estimators=100, max_depth=6, max_features=2, bootstrap=True, criterion="entropy",
    min_samples_split=600, min_samples_leaf=100,
    parameters=["rmag", "rabsmag", "redshift", "rad", "sern"])
classifier.fit(train_galaxies)
classifier.evaluate(train_galaxies), classifier.evaluate(test_galaxies)

#%% Slots
p1 = []
p2 = []
s = (10, 11, 13, 14, 15, 17, 20, 22, 25, 27, 30, 35, 40, 50, 60, 70, 80, 100)

for slots in s:
    print(slots)

    if slots < 40:
        clf = Classifier(slots, n_estimators=100, max_depth=6, max_features=2, bootstrap=True, criterion="entropy")
    else:
        clf = Classifier(slots, n_estimators=10, max_depth=6, max_features=2, bootstrap=True, criterion="entropy")
    
    clf.fit(train_galaxies)
    p1.append(clf.evaluate(test_galaxies))

    q_pdf = PDF.from_samples(np.linspace(0, 1, slots), galaxies["ba"])
    q_slots = test_galaxies["ba"].multiply(slots).apply(np.ceil).astype(int) - 1
    p2.append(np.sum(q_pdf.y[q_slots]) / slots / len(q_slots))

#%%
plt.plot(np.linspace(10, 100, 90), 1/np.array(np.linspace(10, 100, 90)), label="Ühtlase $q$ jaotuse järgi")
plt.plot(s, p2, label="Kogu valimi $q$ jaotuse järgi")
plt.plot(s, p1, label="Treenitud otsustusmetsa järgi")
plt.xlabel("$K$")
plt.ylabel("$\\mathcal{L}$", rotation=0)
plt.legend(frameon=False)

savefig("plots/global_vs_classifier.pdf")

#%%
plt.plot(s[1:], (np.array(p1[1:]) - np.array(p2[1:])) / (np.array(p1[:-1]) - np.array(p2[:-1])))

#%%
p1[10] - p2[10]

#%%
classifier.clf.feature_importances_

#%%
def compare_hist(parameter, cuts, name):
    quantiles = pd.qcut(test_galaxies[parameter], cuts, labels=False)
    median = np.median(test_galaxies[parameter])

    for i in range(len(cuts) - 1):
        hist = np.histogram(test_galaxies[quantiles == i]["ba"].values, classifier.q_slot_multiplier, (0, 1), density=True)[0]
        color = next(plt.gca()._get_lines.prop_cycler)['color']

        if i == 0:
            label = "$%s < %.2f$" % (name, median)
        else:
            label = "$%s > %.2f$" % (name, median)
        
        plt.plot(np.linspace(0, 1, classifier.q_slot_multiplier, endpoint=False) + 1/2/classifier.q_slot_multiplier, hist, 'o', color=color)
        predicted_pdf = np.sum(classifier.clf.predict_proba(
            test_galaxies[quantiles == i][classifier.parameters]
        ), 0) / len(test_galaxies[quantiles == i]) * classifier.q_slot_multiplier
        plt.plot(np.linspace(0, 1, classifier.q_slot_multiplier, endpoint=False) + 1/2/classifier.q_slot_multiplier, predicted_pdf, color=color, label=label)
        
        plt.xlabel("$q$")
        plt.ylabel("$\\rho(q)$", rotation=0, labelpad=22)
        plt.gca().legend(frameon=False)

        savefig("plots/classifier_%s.pdf" % parameter)

#%%
plt.rcParams.update({ 'font.size': 22 })

#%%
compare_hist("sern", (0, 0.5, 1), "\\mathrm{sern}")

#%%
compare_hist("rabsmag", (0, 0.5, 1), "\\mathrm{absmag}_\\mathrm{r}")

#%%
compare_hist("rad", (0, 0.5, 1), "\\mathrm{rad}")

#%%
compare_hist("rmag", (0, 0.5, 1), "\\mathrm{rad}_\\mathrm{r}")

#%%
compare_hist("redshift", (0, 0.5, 1), "\\mathrm{redshift}")

#%%
names = ["A", "B", "C"]

for i, j in enumerate((10, 80, 1210)):
    predicted_pdf = classifier.predict_pdf(test_galaxies.iloc[[j]])
    plt.plot(predicted_pdf.x, predicted_pdf.y, label="Galaktika %s" % names[i])

plt.xlabel("$q$")
plt.ylabel("$\\rho(q)$", rotation=0)
plt.legend(frameon=False)

savefig("plots/random_predicted_pdf.pdf")
