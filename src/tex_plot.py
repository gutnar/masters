#%%
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use("default")

plt.rcParams['text.latex.preamble'] = [r"\usepackage{lmodern}"]
plt.tick_params(direction="in")

plt.rcParams.update({
    'text.usetex': True,
    'font.size': 11,
    'font.family': 'lmodern',
    'text.latex.unicode': True,
})

def savefig(path):
    plt.tick_params(direction="in")
    plt.savefig(path, dpi=1000, bbox_inches='tight')#, pad_inches=0)
