#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from helpers import get_truncnorm_sample

#%%
galaxies = pd.read_csv("data_inclinations.txt")
galaxies.describe()

#%%
x_pdf = np.zeros(100)
x_rand_pdf = np.zeros(100)

for i in range(len(galaxies)):
    galaxy = galaxies.iloc[[i]]
    ba = float(galaxy["ba"])

    x = get_truncnorm_sample(galaxy["x_mu"], galaxy["x_sigma"], 0, ba, 100)
    z = get_truncnorm_sample(galaxy["z_mu"], galaxy["z_sigma"], 0, 1, 100)
    
    x = np.minimum(x, z)

    x2 = x**2
    z2 = z**2

    # sin_p = 0, sin2_p = 0, cos2_p = 1, sin_2p = 0
    #A = (cos2_t/x2 - cos2_t + 1)/z2
    # A = cos2_t*(1/x2 - 1)/z2 + 1/z2
    # B = 0
    # C = 1/x2
    # D = (A - C)

    # (A^2 - A/x2) * ba**2 - A/x2 + (ba/x2)^2 = 0
    # A^2 - (1/ba^2 + 1) / x^2 * A + 1/x^4 = 0

    A = ( (1/ba^2 + 1) + np.sqrt( (1/ba^2 + 1)^2 - 4 ) ) / (2*x^2)


    cos2_t = (A - 1/z2) * z2 / (1/x2 - 1)


    # (A + C + D)**2
    # (A + C)**2 + 3*(A - C)**2
    # A^2 + 2AC + C^2 + 3A^2 - 6AC + 3C^2
    # 4A^2 - 4AC + 4C^2 = 4 (A^2 - AC + C^2)
    
    # (A + C)**2 - (A - C)**2
    # A^2 + 2AC + C^2 - A^2 + 2AC - C^2
    # 4AC

    cos_x = np.sqrt((ba**2 - x**2) / (1 - x**2))
    x_pdf += np.histogram(cos_x, 100, (0, 1), density=True)[0]

    '''
    A = cos2_t/x2 * (sin2_p + cos2_p/z2) + sin2_t/z2
    B = (1 - 1/z2) * 1/x2 * cos_t * sin_2p
    C = (sin2_p/z2 + cos2_p)/x2
    D = np.sqrt((A - C)**2 + B**2)

    ba = np.sqrt((A + C - D) / (A + C + D))
    '''
    
    # Random
    x = get_truncnorm_sample(0.14, 0.1, 0, ba, 100)
    cos_x = np.sqrt((ba**2 - x**2) / (1 - x**2))
    x_rand_pdf += np.histogram(cos_x, 100, (0, 1), density=True)[0]

#x_pdf /= np.sum(x_pdf)
err = np.sum((x_pdf - np.mean(x_pdf))**2)/100/np.mean(x_pdf)**2
err_rand = np.sum((x_rand_pdf - np.mean(x_rand_pdf))**2)/100/np.mean(x_rand_pdf)**2

plt.plot(x_pdf, label="Estimate, e = %.2E" % err)
plt.plot(x_rand_pdf, label="N(0.14, 0.01), e = %.2E" % err_rand)
plt.legend()
#plt.plot(z_pdf)
