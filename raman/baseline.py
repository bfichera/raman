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
