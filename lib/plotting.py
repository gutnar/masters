import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


XZ_MESH = np.meshgrid(
    np.linspace(0, 1, 100), np.linspace(0.5, 1, 100)
)
XZ_GRID = np.append(
    XZ_MESH[0].reshape(-1, 1), XZ_MESH[1].reshape(-1, 1), 1
)

QT_MESH = np.meshgrid(
    np.linspace(0, 1, 100), np.linspace(-np.pi/2, np.pi/2, 100)
)
QT_GRID = np.append(
    QT_MESH[0].reshape(-1, 1), QT_MESH[1].reshape(-1, 1), 1
)

QP_MESH = np.meshgrid(
    np.linspace(0, 1, 100), np.linspace(-np.pi/2, np.pi/2, 100)
)
QP_GRID = np.append(
    QP_MESH[0].reshape(-1, 1), QP_MESH[1].reshape(-1, 1), 1
)

TP_MESH = np.meshgrid(
    np.linspace(-np.pi/2, np.pi/2, 100),
    np.linspace(-np.pi/2, np.pi/2, 100)
)
TP_GRID = np.column_stack((
    TP_MESH[0].reshape(-1, 1),
    TP_MESH[1].reshape(-1, 1)
))


def get_pdf(kde, grid, resample=False, N=10000):
    if resample:
        x, y = kde.resample(N)
        kde = stats.kde.gaussian_kde(
            np.column_stack((x, y)).T
        )
    
    return kde(grid.T).reshape(100, 100)


def plot_ba_results(ba, size=150000):
    q, x, z, p, t = ba.sample(size)
    
    plt.xlim((0, 1))
    plt.plot(ba.q_pdf.x, ba.q_pdf.y, label="target")
    plt.hist(q, 100, (0, 1), True, label=r"$q$", histtype="step")
    plt.hist(np.cos(t), 100, (0, 1), True, label=r"$\cos(\theta)$", histtype="step")
    plt.hist(p/np.pi, 100, (0, 1), True, label=r"$\phi/\pi$", histtype="step")
    plt.gca().legend()


def plot_kde(kde, grid, resample=True, **kwargs):
    pdf = get_pdf(kde, grid, resample)
    plt.imshow(pdf, "magma", origin="lower", **kwargs)


def plot_xz_kde(ba, resample=True, **kwargs):
    plt.xlabel(r"$\xi$")
    plt.xticks(
        [0, 19, 39, 59, 79, 99],
        ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
    )

    plt.ylabel(r"$\zeta$", rotation=0)
    plt.yticks(
        [0, 19, 39, 59, 79, 99],
        [("%.1f" % z) for z in np.linspace(ba.Z_MIN, 1, 6)]
    )

    plot_kde(ba.xz_kde, XZ_GRID, resample, aspect=1/2, **kwargs)


def plot_qt_kde(ba, resample=True, **kwargs):
    plt.xlabel(r"$q$")
    plt.xticks(
        [0, 19, 39, 59, 79, 99],
        ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
    )

    plt.ylabel(r"$\theta$", rotation=0)
    plt.yticks(
        [0, 19, 39, 59, 79, 99],
        [(r"$%d^∘$" % t) for t in np.linspace(-90, 90, 6)]
    )

    plot_kde(ba.qt_kde, QT_GRID, resample, aspect=1, **kwargs)


def plot_qp_kde(ba, resample=True, **kwargs):
    plt.xlabel(r"$q$")
    plt.xticks(
        [0, 19, 39, 59, 79, 99],
        ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
    )

    plt.ylabel(r"$\phi$", rotation=0)
    plt.yticks(
        [0, 19, 39, 59, 79, 99],
        [(r"$%d^∘$" % t) for t in np.linspace(-90, 90, 6)]
    )

    plot_kde(ba.qp_kde, QP_GRID, resample, aspect=1, **kwargs)


def plot_tp_kde(ba, q, **kwargs):
    t, p = ba.sample_tp(q, 10000)

    kde = stats.kde.gaussian_kde(
        np.column_stack((t, p)).T, "scott"
    )

    plt.xlabel(r"$\theta$")
    plt.xticks(
        [0, 19, 39, 59, 79, 99],
        [(r"$%d^∘$" % t) for t in np.linspace(-90, 90, 6)]
    )

    plt.ylabel(r"$\phi$", rotation=0)
    plt.yticks(
        [0, 19, 39, 59, 79, 99],
        [(r"$%d^∘$" % t) for t in np.linspace(-90, 90, 6)]
    )

    plot_kde(kde, TP_GRID, False, **kwargs)
