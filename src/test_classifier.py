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
galaxies = pd.read_csv("data/intermediate/galaxies.csv")
np.random.seed(0b11011101100011011011011001110010)
training_set = np.random.rand(len(galaxies)) < 0.75
train_galaxies = galaxies[training_set]
test_galaxies = galaxies[~training_set]

train_galaxies.to_csv("data/intermediate/train_galaxies.csv", index=False)
test_galaxies.to_csv("data/intermediate/test_galaxies.csv", index=False)

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

    q_pdf = PDF.from_samples(np.linspace(0, 1, slots), galaxies["ba"])
    q_slots = test_galaxies["ba"].multiply(slots).apply(np.ceil).astype(int) - 1
    p2.append(np.sum(q_pdf.y[q_slots]) / slots / len(q_slots))

#%%
plt.plot(np.linspace(10, 100, 90), 1/np.array(np.linspace(10, 100, 90)), label="Ühtlase q jaotuse järgi")
plt.plot(s, p2, label="Kogu valimi q jaotuse järgi")
plt.plot(s, p1, label="Treenitud otsustusmetsa järgi")
plt.legend()

plt.savefig("plots/global_vs_classifier.pdf")

#%%
p1[8] - p2[8]

#%%
classifier.clf.feature_importances_

#%%
def compare_hist(parameter, cuts):
    quantiles = pd.qcut(test_galaxies[parameter], cuts, labels=False)
    median = np.median(test_galaxies[parameter])

    for i in range(len(cuts) - 1):
        hist = np.histogram(test_galaxies[quantiles == i]["ba"].values, 25, (0, 1), density=True)[0]
        color = next(plt.gca()._get_lines.prop_cycler)['color']

        if i == 0:
            label = "%s < %.2f" % (parameter, median)
        else:
            label = "%s > %.2f" % (parameter, median)
        
        plt.plot(np.linspace(0, 1, 25) + 1/50, hist, color=color, label=label)
        predicted_pdf = np.sum(classifier.clf.predict_proba(
            test_galaxies[quantiles == i][classifier.parameters]
        ), 0) / len(test_galaxies[quantiles == i]) * 25
        
        plt.plot(np.linspace(0, 1, 25) + 1/50, predicted_pdf, 'o', color=color)
        
        #plt.title(parameter)
        plt.gca().legend()

        plt.savefig("plots/classifier_%s.pdf" % parameter)

#%%
compare_hist("sern", (0, 0.5, 1))

#%%
compare_hist("rabsmag", (0, 0.5, 1))

#%%
compare_hist("rad", (0, 0.5, 1))

#%%
compare_hist("rmag", (0, 0.5, 1))

#%%
compare_hist("redshift", (0, 0.5, 1))

#%%
for i in (10, 80, 190):
    predicted_pdf = classifier.predict_pdf(test_galaxies.iloc[[i]])
    plt.plot(predicted_pdf.x, predicted_pdf.y)

plt.savefig("plots/random_predicted_pdf.pdf")
