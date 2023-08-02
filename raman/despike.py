import numpy as np


def despike(x, y, ignore=[], threshold=12):

    ma = 3

    def modified_z_score(y):
        yb = np.diff(y)
        median_y = np.median(yb)
        median_absolute_deviation_y = np.median(
            [np.abs(y - median_y) for y in yb],
        )
        modified_z_scores = [
            0.6745 * (y - median_y) / median_absolute_deviation_y
            for y in yb
        ]
        return modified_z_scores

    def fixer(x, y, ma):
        spikes = (abs(np.array(modified_z_score(y))) > threshold)
        ignoreit = [
            True in [
                xi > r[0] and xi < r[1]
                for r in ignore
            ]
            for xi in x
        ]
        spikes = [(s and not ig) for s, ig in zip(spikes, ignoreit)]
        y_out = y.copy()
        for i in np.arange(len(spikes)):
            if spikes[i] != 0:
                w = np.arange(max(i-ma, 0), min(i+1+ma, len(spikes)))
                we = []
                for wi in w:
                    if spikes[wi] == 0:
                        we.append(wi)
                y_out[i] = np.mean(y.take(we))
        return y_out

    return fixer(x, y, ma)
