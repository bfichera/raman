import numpy as np
import polars as pl
import matplotlib.pyplot as plt


class ModeData:

    def __init__(self, center_frequency, left_bound, right_bound, _modedata):
        self.center_frequency = center_frequency
        self.left_bound = left_bound
        self.right_bound = right_bound
        self._modedata = _modedata
        self.flattened_pdata = _modedata.flattened_pdata
        self.flattened_adata = _modedata.flattened_adata
        self.flattened_ydata = _modedata.flattened_ydata
        self.a_diff_angles = _modedata.a_diff_angles

    @property
    def bounds(self):
        return (self.left_bound, self.right_bound)

    @property
    def width(self):
        return self.right_bound - self.left_bound

    def pdata_of(self, a_diff):
        return self._modedata.pdata_of(a_diff)

    def ydata_of(self, a_diff):
        return self._modedata.ydata_of(a_diff)

    def sample_ydata(self):
        return self._modedata.sample_ydata()

    def plot(self):
        self._modedata.plot()


class _ModeData:

    def __init__(self, pdatas, adatas, ydatas):
        pdatas = []
        adatas = []
        ydatas = []
        p_angles = np.arange(0, 360, 10)
        for a_ in [0, 90]:
            pdata = p_angles
            adata = p_angles + a_
            sin = np.sin
            cos = np.cos
            a = 3.21
            d = 1.45
            a_angle = adata
            p_angle = pdata
            pi = np.pi
            ydata = 2*a**2*sin(pi*a_angle/180)**2*sin(pi*p_angle/180)**2 - a**2*sin(pi*a_angle/180)**2 + a**2*sin(pi*a_angle/90)*sin(pi*p_angle/90)/2 - a**2*sin(pi*p_angle/180)**2 + a**2 + 2*a*d*sin(pi*a_angle/180)*cos(pi*a_angle/180) + 2*a*d*sin(pi*p_angle/180)*cos(pi*p_angle/180) - 2*d**2*sin(pi*a_angle/180)**2*sin(pi*p_angle/180)**2 + d**2*sin(pi*a_angle/180)**2 + d**2*sin(pi*a_angle/90)*sin(pi*p_angle/90)/2 + d**2*sin(pi*p_angle/180)**2
            pdatas.append(pdata)
            adatas.append(adata)
            ydatas.append(ydata)
        pdatas = np.array(pdatas)
        adatas = np.array(adatas)
        ydatas = np.array(ydatas)
        self.a_diff_angles = np.unique(adatas - pdatas)

        self.df = pl.concat(
            [
                pl.LazyFrame(
                    {
                        'P_ANGLE': p,
                        'A_ANGLE': a,
                        'COUNTS/SEC': y,
                    },
                )
                for p, a, y in zip(pdatas, adatas, ydatas)
            ],
            how='vertical',
        ).with_columns(
            (pl.col('A_ANGLE') - pl.col('P_ANGLE')).alias('A_DIFF_ANGLE'),
        ).collect()
        self.flattened_pdata = pdatas.flatten()
        self.flattened_adata = adatas.flatten()
        self.flattened_ydata = ydatas.flatten()

    def pdata_of(self, a):
        return np.array(
            self.df
            .filter(pl.col('A_DIFF_ANGLE') == a)
            .select(pl.col('P_ANGLE'))
            .to_series()
        )

    def ydata_of(self, a):
        return np.array(
            self.df
            .filter(pl.col('A_DIFF_ANGLE') == a)
            .select(pl.col('COUNTS/SEC'))
            .to_series()
        )

    def sample_ydata(self):
        return np.array(
            self.df
            .filter(pl.col('A_DIFF_ANGLE') == self.a_diff_angles[0])
            .select(pl.col('COUNTS/SEC'))
            .to_series()
        )

    def plot(self):
        for a in self.a_diff_angles:
            plt.plot(
                self.pdata_of(a),
                self.ydata_of(a),
                label='$a='+str(a)+r'^\circ$',
            )
        plt.legend()
        plt.show()
