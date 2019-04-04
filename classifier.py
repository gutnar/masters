#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from helpers import PDF
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

q_slot_multiplier = 30

#%% Load data
galaxies = pd.read_csv("data/raw/data_gama_gal_orient.txt", r"\s+")
galaxies["baslot"] = galaxies["ba"].multiply(q_slot_multiplier).apply(np.ceil).astype(int) - 1

galaxies.describe()

#%% Generate training and test sets
np.random.seed(0b11011100000011011011000001110000)
training_set = np.random.rand(len(galaxies)) < 0.75
galaxies_train = galaxies[training_set]
galaxies_test = galaxies[~training_set]

#%% Train classifier
parameters = ["rmag", "rabsmag", "redshift", "rad", "sern"]

clf = RandomForestClassifier(
    criterion="gini",
    n_estimators=64,
    #max_depth=10,
    min_samples_split=10,
    min_samples_leaf=25,
    max_features=None,
    n_jobs=-1
)

#clf = KNeighborsClassifier(10)

X_train = galaxies_train[parameters].values
Y_train = galaxies_train["baslot"].values

X_test = galaxies_test[parameters].values
Y_test = galaxies_test["baslot"].values

clf.fit(X_train, Y_train)

#%%
def predict_pdf(galaxy):
    pdf = clf.predict_proba(galaxy[parameters].values)[0] * q_slot_multiplier
    return PDF(pdf, np.linspace(0, 1, q_slot_multiplier) + 1/q_slot_multiplier/2)

#%%
def compare_hist(galaxies, parameter, cuts):
    quantiles = pd.qcut(galaxies[parameter], cuts, labels=False)

    for i in range(len(cuts) - 1):
        hist = np.histogram(galaxies[quantiles == i]["baslot"].values, q_slot_multiplier, (0, q_slot_multiplier))[0]
        color = next(plt.gca()._get_lines.prop_cycler)['color']
        
        plt.plot(hist, 'o', color=color, label="("+str(round(cuts[i], 2))+", "+str(round(cuts[i+1], 2))+"]")
        plt.plot(np.sum(clf.predict_proba(galaxies[quantiles == i][parameters].values), 0), color=color)
        
        plt.title(parameter)
        plt.gca().legend()


#%%
if __name__ == '__main__':
    plt.title("Training set performance")
    plt.plot(np.sum(clf.predict_proba(X_train), 0))
    plt.plot(np.histogram(Y_train, q_slot_multiplier, (0, q_slot_multiplier))[0], 'o')

#%%
if __name__ == '__main__':
    plt.title("Test set performance")
    plt.plot(np.sum(clf.predict_proba(X_test), 0))
    plt.plot(np.histogram(Y_test, q_slot_multiplier, (0, q_slot_multiplier))[0], 'o')

#%%
if __name__ == '__main__':
    compare_hist(galaxies_test, "rmag", (0, 1/2, 1))

#%%
if __name__ == '__main__':
    compare_hist(galaxies_test, "rabsmag", (0, 1/2, 1))

#%%
if __name__ == '__main__':
    compare_hist(galaxies_test, "redshift", (0, 1/2, 1))

#%%
if __name__ == '__main__':
    compare_hist(galaxies_test, "rad", (0, 1/2, 1))

#%%
if __name__ == '__main__':
    compare_hist(galaxies_test, "sern", (0, 1/2, 1))
