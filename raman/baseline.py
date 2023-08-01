import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import sparse
from scipy.sparse.linalg import spsolve


def baseline(y, lam, p, N=10):
    m = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(m, m-2))
    w = np.ones(m)
    for i in tqdm(range(N)):
        W = sparse.spdiags(w, 0, m, m)
        Z = W + lam * (D @ D.transpose())
        z = spsolve(Z, W @ y)
        w = p * (y > z).astype(int) + (1 - p) * (y < z).astype(int)
    return z


def baseline_interactive(y):
    while True:
        lam = float(input('lambda: '))
        p = float(input('p: '))
        b = baseline(y, lam, p)
        plt.plot(y)
        plt.plot(b)
        plt.show()


if __name__ == '__main__':
    def Gauss(x, mu, sigma, A = 1):
        # This def returns the Gaussian function of x
        # x is an array
        # mu is the expected value
        # sigma is the square root of the variance
        # A is a multiplication factor
        
        gaussian = A/(sigma * np.sqrt(2*np.pi)) * np.exp(-0.5*((x-mu)/sigma)**2)
        
        return gaussian

    # X-axis (Wavelengths)
    x_range =  np.linspace(650, 783, 1024)

    # Let's create three different components

    # Component A
    mu_a1 = 663
    sigma_a1 = 1
    intensity_a1 = 1

    mu_a2 = 735
    sigma_a2 = 1
    intensity_a2 = 0.2

    mu_a3 = 771
    sigma_a3 = 1
    intensity_a3 = 0.3

    gauss_a =  Gauss(x_range, mu_a1, sigma_a1, intensity_a1) + Gauss(x_range, mu_a2, sigma_a2, intensity_a2) + Gauss(x_range, mu_a3, sigma_a3, intensity_a3)

    # Component B
    mu_b = 700
    sigma_b = 1
    intensity_b = 0.2

    mu_b1 = 690
    sigma_b1 = 2
    intensity_b1 = 0.5

    mu_b2 = 710
    sigma_b2 = 1
    intensity_b2 = 0.75

    mu_b3 = 774
    sigma_b3 = 1.5
    intensity_b3 = 0.25

    gauss_b = Gauss(x_range, mu_b, sigma_b, intensity_b) + Gauss(x_range, mu_b1, sigma_b1, intensity_b1) + Gauss(x_range, mu_b2, sigma_b2, intensity_b2) + Gauss(x_range, mu_b3, sigma_b3, intensity_b3)

    # Component C
    mu_c1 = 660
    sigma_c1 = 1
    intensity_c1 = 0.05

    mu_c2 = 712
    sigma_c2 = 4
    intensity_c2 = 0.7

    gauss_c = Gauss(x_range, mu_c1, sigma_c1, intensity_c1) + Gauss(x_range, mu_c2, sigma_c2, intensity_c2)

    # Component normalization
    component_a = gauss_a/np.max(gauss_a)
    component_b = gauss_b/np.max(gauss_b)
    component_c = gauss_c/np.max(gauss_c)

    # What concentrations we want these components to have in our mixture:
    c_a = 0.5
    c_b = 0.3
    c_c = 0.2

    comps = np.array([c_a, c_b, c_c])

    # Let's build the spectrum to be studied: The mixture spectrum
    mix_spectrum = c_a * component_a + c_b * component_b + c_c *component_c

    # Let's add some noise for a bit of realism:

    # Random noise:
    mix_spectrum = mix_spectrum +  np.random.normal(0, 0.02, len(x_range))

    # Spikes: 
    mix_spectrum[800] = mix_spectrum[800] + 1
    mix_spectrum[300] = mix_spectrum[300] + 0.3

    # Baseline as a polynomial background:
    poly = 0.2 * np.ones(len(x_range)) + 0.0001 * x_range + 0.000051 * (x_range - 680)**2 
    mix_spectrum = mix_spectrum + poly

    baseline_interactive(mix_spectrum)
