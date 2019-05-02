import numpy as np

n_clusters = 2

galaxy_classes = (
    { "label": "e_spiral", "parameter": "e_class", "value": 0 },
    { "label": "e_elliptic", "parameter": "e_class", "value": 1 },
    { "label": "g_spiral", "parameter": "g_class", "value": 0 },
    { "label": "g_elliptic", "parameter": "g_class", "value": 1 },
    #{ "label": "g_unknown1", "parameter": "g_class", "value": 2 },
    #{ "label": "g_unknown2", "parameter": "g_class", "value": 3 }
)

dum_bins = np.linspace(0, 1, 101)
