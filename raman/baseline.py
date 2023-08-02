import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt


def baseline_noninteractive(y, lam, p, niter=100, exclude=[]):
    y_ = y.copy()
    for r1, r2 in exclude:
        for i in np.arange(r1, r2):
            y_[i] = (y_[r2] - y_[r1])*(i-r1)/(r2-r1)+y_[r1]
    m = len(y_)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(m, m-2))
    w = np.ones(m)
    for i in range(niter):
        W = sparse.spdiags(w, 0, m, m)
        Z = W + lam * (D @ D.transpose())
        z = spsolve(Z, W @ y_)
        w = p * (y_ > z).astype(int) + (1 - p) * (y_ < z).astype(int)
    return z


def baseline(y, lam=None, p=None, niter=100, exclude=[], interactive=False, xdata=None):
    smoothed = savgol_filter(y, 11, 3)
    if interactive is False:
        return baseline_noninteractive(smoothed, lam, p, niter, exclude)
    while True:
        try:
            lam = float(input('lambda: '))
            p = float(input('p: '))
            b = baseline_noninteractive(smoothed, lam, p, niter, exclude)
            if xdata is None:
                xdata = np.arange(len(smoothed))
            plt.plot(xdata, y)
            plt.plot(xdata, b)
            plt.show()
        except:
            return b


def baseline_interactive(*args, **kwargs):
    kwargs['interactive'] = True
    return baseline(*args, **kwargs)
