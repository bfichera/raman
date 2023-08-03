import numpy as np
import polars as pl
import matplotlib.pyplot as plt


class Mode:

    def __init__(self, center_frequency, left_bound, right_bound, modedata):
        self.center_frequency = center_frequency
        self.left_bound = left_bound
        self.right_bound = right_bound
        self.modedata = modedata

    @property
    def bounds(self):
        return (self.left_bound, self.right_bound)

    @property
    def width(self):
        return self.right_bound - self.left_bound

    def pdata_of(self, a):
        return self.modedata.pdata_of(a)

    def ydata_of(self, a):
        return self.modedata.ydata_of(a)

    def plot(self):
        self.modedata.plot()


class _ModeData:

    def __init__(self, pdatas, ydatas, a_angles):
        self.a_angles = a_angles
        self.df = pl.concat(
            [
                pl.LazyFrame(
                    {
                        'P_ANGLE': p,
                        'A_ANGLE': a,
                        'COUNTS/SEC': y,
                    },
                )
                for p, y, a in zip(pdatas, ydatas, a_angles)
            ],
            how='vertical',
        ).collect()

    def pdata_of(self, a):
        return np.array(
            self.df
            .filter(pl.col('A_ANGLE') == a)
            .select(pl.col('P_ANGLE'))
            .to_series()
        )

    def ydata_of(self, a):
        return np.array(
            self.df
            .filter(pl.col('A_ANGLE') == a)
            .select(pl.col('COUNTS/SEC'))
            .to_series()
        )

    def plot(self):
        for a in self.a_angles:
            plt.plot(self.pdata_of(a), self.ydata_of(a), label='$a='+str(a)+r'^\circ$')
        plt.legend()
        plt.show()
