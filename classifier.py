#%%
from common import pd, np, plt, galaxies_train, galaxies_test, parameters
from sklearn.ensemble import RandomForestClassifier


def compare_hist(galaxies, parameter, cuts):
    quantiles = pd.qcut(galaxies[parameter], cuts, labels=False)

    for i in range(len(cuts) - 1):
        hist = np.histogram(galaxies[quantiles == i]["baslot"].values, 100, (0, 100))[0]
        color = next(plt.gca()._get_lines.prop_cycler)['color']
        
        plt.plot(hist, 'o', color=color, label="("+str(round(cuts[i], 2))+", "+str(round(cuts[i+1], 2))+"]")
        plt.plot(np.sum(clf.predict_proba(galaxies[quantiles == i][parameters].values), 0), color=color)
        
        plt.title(parameter)
        plt.gca().legend()

#%%
clf = RandomForestClassifier(
    n_estimators=64,
    #max_depth=10,
    #min_samples_split=5,
    min_samples_leaf=50,
    max_features=None,
    n_jobs=-1
)

X_train = galaxies_train[parameters].values
Y_train = galaxies_train["baslot"].values

X_test = galaxies_test[parameters].values
Y_test = galaxies_test["baslot"].values

clf.fit(X_train, Y_train)

#%%
if __name__ == '__main__':
    plt.title("Training set performance")
    plt.plot(np.sum(clf.predict_proba(X_train), 0))
    plt.plot(np.histogram(Y_train, 100, (0, 100))[0], 'o')

#%%
if __name__ == '__main__':
    plt.title("Test set performance")
    plt.plot(np.sum(clf.predict_proba(X_test), 0))
    plt.plot(np.histogram(Y_test, 100, (0, 100))[0], 'o')

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
