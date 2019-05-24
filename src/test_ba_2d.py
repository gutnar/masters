#%%
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm

from lib import PDF, Classifier, BayesianApproximation2d
from lib.plotting import *
from src.tex_plot import savefig


def get_truncnorm_sample(mu, sigma, a, b, N):
    return stats.truncnorm.rvs((a - mu)/sigma, (b - mu)/sigma, loc=mu, scale=sigma, size=N)

#%%
galaxies = pd.read_csv("data/intermediate/galaxies.csv")
#train_galaxies = pd.read_csv("data/intermediate/train_galaxies.csv")
#test_galaxies = pd.read_csv("data/intermediate/test_galaxies.csv")

spiral = galaxies[galaxies["sern"] < 2]
elliptic = galaxies[galaxies["sern"] > 2]

spiral_pdf = PDF.from_samples(
    np.linspace(0, 1, 100),
    spiral["ba"]
)

elliptic_pdf = PDF.from_samples(
    np.linspace(0, 1, 100),
    elliptic["ba"]
)

#plt.hist(galaxies["ba"].values, 100, (0, 1), density=True)
plt.plot(spiral_pdf.x, spiral_pdf.y)
plt.plot(elliptic_pdf.x, elliptic_pdf.y)

#%% Manual fit
#classifier = Classifier()
#classifier.fit(train_galaxies)

#%%
spiral_ba = BayesianApproximation2d(spiral_pdf)
spiral_ba.run()

plot_results(spiral_ba)

#%%
elliptic_ba = BayesianApproximation2d(elliptic_pdf)
elliptic_ba.run()

plot_results(elliptic_ba)

#%%
def plot_results(ba, label, size=150000):
    q, xi, zeta, theta, phi = ba.sample(size)
    
    q_kde = sm.nonparametric.KDEUnivariate(q)
    q_kde.fit(bw=0.03)
    q_pdf = q_kde.evaluate(ba.q_pdf.x)
    
    color = next(plt.gca()._get_lines.prop_cycler)['color']

    plt.xlabel("$q$")
    plt.ylabel("$\\rho(q)$", rotation=0, labelpad=15)
    plt.plot(ba.q_pdf.x, ba.q_pdf.y, "x", alpha=0.5, color=color)
    plt.plot(ba.q_pdf.x, q_pdf, label=label, color=color)

#%%
plot_results(spiral_ba, "Spiraalsed galaktikad")
plot_results(elliptic_ba, "Elliptilised galaktikad")

plt.legend(frameon=False)

savefig("plots/spiral_elliptic_fits.pdf")

#%% cos(theta) distribution
def plot_ba_2d_results3(ba, label, size=150000):
    q, xi, zeta, theta, phi = ba.sample(size)

    cos_theta = np.cos(theta)
    cos_theta_kde = sm.nonparametric.KDEUnivariate(cos_theta)
    cos_theta_kde.fit(bw=0.03)
    cos_theta_pdf = cos_theta_kde.evaluate(np.linspace(-0.5, 0.5, 100)) * 2
    
    color = next(plt.gca()._get_lines.prop_cycler)['color']

    plt.plot(np.linspace(0, 1, 100), cos_theta_pdf, color=color, label=label)


plt.rcParams.update({ 'font.size': 22 })
plt.ylim((0.9, 1.1))
plt.xlabel("$\\cos\\theta$")
plt.ylabel("$\\rho(\\cos\\theta)$", rotation=0, labelpad=35)
plot_ba_2d_results3(spiral_ba, "Spiraalsed galaktikad")
plot_ba_2d_results3(elliptic_ba, "Elliptilised galaktikad")
plt.legend(frameon=False)

savefig("plots/spiral_elliptic_rho_cos_theta.pdf")

#%% phi distribution
def plot_ba_2d_results4(ba, label, size=150000):
    q, xi, zeta, theta, phi = ba.sample(size)

    phi_kde = sm.nonparametric.KDEUnivariate(phi)
    phi_kde.fit(bw=0.03)
    phi_pdf = phi_kde.evaluate(np.linspace(-np.pi/2, np.pi/2, 100)) * 2 * np.pi
    
    color = next(plt.gca()._get_lines.prop_cycler)['color']

    plt.plot(np.linspace(-np.pi/2, np.pi/2, 100), phi_pdf, color=color, label=label)

plt.rcParams.update({ 'font.size': 22 })
plt.ylim((0.9, 1.1))
plt.xlabel("$\\phi$")
plt.ylabel("$\\rho(\\phi)$", rotation=0, labelpad=35)
plot_ba_2d_results4(spiral_ba, "Spiraalsed galaktikad")
plot_ba_2d_results4(elliptic_ba, "Elliptilised galaktikad")
plt.legend(frameon=False)

savefig("plots/spiral_elliptic_rho_phi.pdf")

#%%
plot_xz_kde(spiral_ba, False)
savefig("plots/xi_zeta_spiral.pdf")

#%%
plot_xz_kde(elliptic_ba, False)
savefig("plots/xi_zeta_elliptic.pdf")


#%%
train_galaxies = pd.read_csv("data/intermediate/train_galaxies.csv")
test_galaxies = pd.read_csv("data/intermediate/test_galaxies.csv")

len(train_galaxies), len(test_galaxies)

#%% Manual fit
classifier = Classifier()
classifier.fit(train_galaxies)
classifier.evaluate(train_galaxies), classifier.evaluate(test_galaxies)

#%%
def plot_inc_distribution(ba, q, label, **plot):
    pos, inc = ba.sample_pos_inc(q, 0)

    kde = sm.nonparametric.KDEUnivariate(inc)
    kde.fit(bw=0.03)
    pdf = kde.evaluate(np.linspace(0, 1, 100)) * 2

    plt.xlim((0, 1))
    plt.xlabel("$\\cos\\theta$")
    plt.ylabel("$\\rho(\\cos\\theta)$", rotation=0, labelpad=22)
    plt.plot(np.linspace(0, 1, 100), pdf, label=label, **plot)


def plot_pos_distribution(ba, q, p, label, **plot):
    pos, inc = ba.sample_pos_inc(q, p)

    kde = sm.nonparametric.KDEUnivariate(pos)
    kde.fit(bw=0.1)
    pdf = kde.evaluate(np.linspace(-np.pi/2, np.pi/2, 100)) * 2

    plt.xlim((-np.pi/2, np.pi/2))
    plt.xlabel("$\\phi$ [rad]")
    plt.ylabel("$\\rho(\\phi)$", rotation=0, labelpad=22)
    plt.plot(np.linspace(-np.pi/2, np.pi/2, 100), pdf, label=label, **plot)

#%%
plot_inc_distribution(spiral_ba, 0.1, "Spiraalne, $q = 0,1$ ja $p = 0$", linestyle="--", color="#1f77b4")
plot_inc_distribution(elliptic_ba, 0.1, "Elliptiline, $q = 0,1$ ja $p = 0$", linestyle=":", color="#1f77b4")
plot_inc_distribution(spiral_ba, 0.5, "Spiraalne, $q = 0,5$ ja $p = 0$", linestyle="--", color="#ff7f0e")
plot_inc_distribution(elliptic_ba, 0.5, "Elliptiline, $q = 0,5$ ja $p = 0$", linestyle=":", color="#ff7f0e")
plot_inc_distribution(spiral_ba, 0.9, "Spiraalne, $q = 0,9$ ja $p = 0$", linestyle="--", color="#2ca02c")
plot_inc_distribution(elliptic_ba, 0.9, "Elliptiline, $q = 0,9$ ja $p = 0$", linestyle=":", color="#2ca02c")

plt.legend(frameon=False)
savefig("plots/spiral_elliptic_inc.pdf")

#%%
plot_pos_distribution(spiral_ba, 0.1, 0, "Spiraalne, $q = 0,1$ ja $p = 0$", linestyle="--", color="#1f77b4")
plot_pos_distribution(elliptic_ba, 0.1, 0, "Elliptiline, $q = 0,1$ ja $p = 0$", linestyle=":", color="#1f77b4")
plot_pos_distribution(spiral_ba, 0.5, 0, "Spiraalne, $q = 0,5$ ja $p = 0$", linestyle="--", color="#ff7f0e")
plot_pos_distribution(elliptic_ba, 0.5, 0, "Elliptiline, $q = 0,5$ ja $p = 0$", linestyle=":", color="#ff7f0e")
plot_pos_distribution(spiral_ba, 0.9, 0, "Spiraalne, $q = 0,9$ ja $p = 0$", linestyle="--", color="#2ca02c")
plot_pos_distribution(elliptic_ba, 0.9, 0, "Elliptiline, $q = 0,9$ ja $p = 0$", linestyle=":", color="#2ca02c")

plt.legend(frameon=False)
savefig("plots/spiral_elliptic_pos.pdf")

#%%
names = ["A", "B", "C"]
abc_ba = []

for i, j in enumerate((10, 80, 1210)):
    predicted_pdf = classifier.predict_pdf(test_galaxies.iloc[[j]])
    ba = BayesianApproximation2d(predicted_pdf)
    ba.run()
    abc_ba.append(ba)

    plot_results(ba, "Galaktika %s" % names[i])

plt.xlabel("$q$")
plt.ylabel("$\\rho(q)$", rotation=0)
plt.legend(frameon=False)

savefig("plots/random_fits.pdf")

#%%
for i, j in enumerate((10, 80, 1210)):
    color = next(plt.gca()._get_lines.prop_cycler)['color']

    plot_inc_distribution(
        abc_ba[0],
        float(test_galaxies.iloc[[j]]["ba"]),
        "Galaktika %s" % names[i],
        color=color
    )

    plot_inc_distribution(
        spiral_ba,
        float(test_galaxies.iloc[[j]]["ba"]),
        None,
        color=color,
        linestyle="--",
        alpha=0.5
    )

    plot_inc_distribution(
        elliptic_ba,
        float(test_galaxies.iloc[[j]]["ba"]),
        None,
        color=color,
        linestyle=":",
        alpha=0.5
    )

plt.legend(frameon=False)
savefig("plots/random_inc.pdf")

#%%
for i, j in enumerate((10, 80, 1210)):
    color = next(plt.gca()._get_lines.prop_cycler)['color']

    plot_pos_distribution(
        abc_ba[0],
        float(test_galaxies.iloc[[j]]["ba"]),
        float(test_galaxies.iloc[[j]]["pos"])/180*np.pi,
        "Galaktika %s" % names[i],
        color=color
    )

    plot_pos_distribution(
        spiral_ba,
        float(test_galaxies.iloc[[j]]["ba"]),
        float(test_galaxies.iloc[[j]]["pos"])/180*np.pi,
        None,
        color=color,
        linestyle="--",
        alpha=0.5
    )

    plot_pos_distribution(
        elliptic_ba,
        float(test_galaxies.iloc[[j]]["ba"]),
        float(test_galaxies.iloc[[j]]["pos"])/180*np.pi,
        None,
        color=color,
        linestyle=":",
        alpha=0.5
    )

plt.legend(frameon=False)
savefig("plots/random_pos.pdf")

#%%
test_galaxies.iloc[[10, 80, 1210]]
