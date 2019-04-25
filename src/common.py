import numpy as np

n_clusters = 4

galaxy_classes = (
    { "label": "e_spiral", "parameter": "e_class", "value": 0 },
    { "label": "e_elliptic", "parameter": "e_class", "value": 1 },
    { "label": "g_spiral", "parameter": "g_class", "value": 0 },
    { "label": "g_elliptic", "parameter": "g_class", "value": 1 },
    { "label": "g_spiral2", "parameter": "g_class", "value": 2 },
    { "label": "g_elliptic2", "parameter": "g_class", "value": 3 }
)

dum_bins = np.linspace(0, 1, 101)
