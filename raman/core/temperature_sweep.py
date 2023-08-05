import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from scipy.signal import medfilt
import pickle

from .polarization_sweep import PolarizationSweepData


def normalizer(vmin, vmax):

    def func(x):
        return (x-vmin)/(vmax - vmin)

    return func


def load_polarization_sweep(path):
    with open(path, 'rb') as fh:
        return pickle.load(fh)


class TemperatureSweepData:
    """Collection of counts vs. wavenumber for different polarizations"""
    # TODO need to figure out what to do if the x axes are different
    def __init__(
        self,
        temperatures,
        p_angles,
        a_diff_angles,
        paths,
    ):
        self._dict = {}
        self._pols = []

        for p in p_angles:
            for a in a_diff_angles:
                if (p, a) not in self._pols:
                    self._pols.append((p, a))

        for t in np.unique(temperatures):
            p_angles_ = []
            a_diff_angles_ = []
            paths_ = []
            for t_, p, a, path in enumerate(
                zip(temperatures, p_angles, a_diff_angles, paths),
            ):
                if t_ == t:
                    p_angles_.append(p)
                    a_diff_angles_.append(a)
                    paths_.append(path)
            self._dict[t] = PolarizationSweepData(
                p_angles_,
                a_diff_angles_,
                paths_,
            )
        self._temperatures = np.unique(temperatures)

    @property
    def _df(self):
        dfs = []
        for t in self._temperatures:
            dfs.append(
                self._dict[t]._df.with_columns(
                    pl.lit(t).alias('TEMPERATURE'),
                )
            )
        return pl.concat(dfs)

    @property
    def xdata(self):
        return self._xdata_of(
            self._df,
            (pl.col('P_ANGLE') == self.p_angles[0])
            & (pl.col('A_DIFF_ANGLE') == self.a_diff_angles[0])
            & (pl.col('TEMPERATURE') == self._temperatures[0]),
        )

    def _xdata_of(self, filter):
        return np.array(
            self._df
            .filter(filter)
            .select(pl.col('WAVENUMBER'))
            .to_series()
        )

    def _ydata_of(self, filter):
        return np.array(
            self._df
            .filter(filter)
            .select(pl.col('COUNTS/SEC'))
            .to_series()
        )

    def _waterfall_plot(self, offset_factor, cmap=None):
        if cmap is None:
            cmap = lambda x: colormaps['magma'](x*0.7)
        fig, axd = plt.subplot_mosaic(
            [self._pols],
            sharex=True,
            sharey=True,
        )
        for pi, pol in self._pols:
            p, a = pol
            for t in self._temperatures:
                xdata = self.xdata
                filt = (
                    (pl.col('P_ANGLE') == p)
                    & (pl.col('A_DIFF_ANGLE') == a)
                    & (pl.col('TEMPERATURE') == t)
                )
                ydata = self._ydata_of(
                    filt,
                )
                norm_t = normalizer(
                    min(self._temperatures),
                    max(self._temperatures),
                )(t)
                offset = norm_t * offset_factor
                axd[pol].plot(xdata, ydata+offset, color=cmap(norm_t))
                axd[pol].text(
                    xdata[-1],
                    (ydata+offset)[-1],
                    f'{int(t)} K',
                )
        for pol in self._pols:
            axd[pol].set_xlabel(r'$\nu$ (cm${}^{-1}$)')
            axd[pol].set_ylabel(r'counts $\cdot$ s${}^{-1}$')
            axd[pol].set_title(
                '$p = '+str(p)+r'^\circ$, $a = '+str(a)+r'^\circ$',
            )
        return fig, axd

    def waterfall(self, offset_factor=0, cmap=None):
        fig, axd = self._waterfall_plot(offset_factor, cmap=cmap)
        return fig, axd

    def pcolor(self):
        fig, axd = plt.subplot_mosaic(
            [self._pols],
            sharex=True,
            sharey=True,
        )
        for pol in self.pols:
            p, a = pol
            ydatas = []
            ax = axd[pol]
            all_maxes = []
            for t in self._temperatures:
                filt = (
                    (pl.col('TEMPERATURE') == t)
                    & (pl.col('P_ANGLE') == p)
                    & (pl.col('A_DIFF_ANGLE') == a)
                )
                ydata = self._ydata_of(filt)
                all_maxes.append(max(medfilt(ydata, 11)))
                ydatas.append(ydata)
            ydatas = np.array(ydatas)
            X, Y = np.meshgrid(self._temperatures, self.xdata)
            c = ax.pcolor(
                X,
                Y,
                ydatas.transpose(),
                vmin=0,
                vmax=max(all_maxes),
                cmap='magma',
                shading='nearest',
            )
            fig.colorbar(c, ax=ax)
            ax.set_title(r'$p='+str(p)+r'^\circ$, $a='+str(a)+r'^\circ$')
            ax.set_xlabel(r'$\nu~(\mathrm{cm}^{-1})$')
            ax.set_ylabel(r'$T$ (K)')
        plt.show()

    def check_despike(self, offset_factor=0):
        for t, d in self._dict.items():
            d.check_despike(offset_factor=offset_factor)

    def check_background(self, offset_factor=0):
        for t, d in self._dict.items():
            d.check_background(offset_factor=offset_factor)

    def check_baseline(self, offset_factor=0):
        for t, d in self._dict.items():
            d.check_baseline(offset_factor=offset_factor)

    def despike(self, ignore=[], threshold=12):
        for t, d in self._dict.items():
            d.despike(ignore=ignore, threshold=threshold)

    def subtract_background(
        self,
        background_paths,
        background_a_diff_angles,
        background_reference_peak,
    ):
        for t, d in self._dict.items():
            d.subtract_background(
                background_paths,
                background_a_diff_angles,
                background_reference_peak,
            )

    def subtract_baseline(self, lam, p, niter=100, exclude=[],
                          baseline=None, interactive=False):
        for t, d in self._dict.items():
            d.subtract_baseline(
                lam,
                p,
                niter=niter,
                exclude=exclude,
                baseline=baseline,
                interactive=interactive,
            )

    def to_pickle(self, path):
        with open(path, 'wb') as fh:
            pickle.dump(self, fh)


class _Baseline:

    def __init__(self, df):
        self.df = df
