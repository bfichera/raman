import polars as pl
import numpy as np
from scipy.signal import find_peaks, peak_prominences
import matplotlib.pyplot as plt
import pickle

from ..despike import despike
from ..baseline import baseline
from ..mode import ModeData, _ModeData


def normalizer(vmin, vmax):

    def func(x):
        return (x-vmin)/(vmax - vmin)

    return func


def load_polarization_sweep(path):
    with open(path, 'rb') as fh:
        return pickle.load(fh)


class PolarizationSweepData:
    """Collection of counts vs. wavenumber for different polarizations"""
    # TODO need to figure out what to do if the x axes are different
    def __init__(
        self,
        p_angles,
        a_diff_angles,
        paths,
    ):
        p_angles = p_angles.astype(int)
        a_diff_angles = a_diff_angles.astype(int)
        self.p_angles = np.unique(p_angles)
        self.a_diff_angles = np.unique(a_diff_angles)
        self._all_p_angles = p_angles
        self._all_a_diff_angles = a_diff_angles
        self._back_df = None
        self._normalized_back_df = None
        self._back_sub_df = None
        self._despiked_df = None
        self._baseline_df = None
        self._base_sub_df = None
        self._raw_df = pl.concat(
            [
                pl.scan_csv(
                    path,
                    comment_char='#',
                    encoding='utf8-lossy',
                    separator='\t',
                    has_header=False,
                    new_columns=['WAVENUMBER', 'COUNTS/SEC'],
                ).with_columns(
                    pl.lit(p).alias('P_ANGLE'),
                    pl.lit(a).alias('A_DIFF_ANGLE'),
                ).sort(
                    pl.col('WAVENUMBER'),
                )
                for p, a, path in zip(p_angles, a_diff_angles, paths)
            ],
            how='vertical',
        ).collect()

    @property
    def _df(self):
        if self._base_sub_df is not None:
            return self._base_sub_df
        if self._despiked_df is not None:
            return self._despiked_df
        if self._back_sub_df is not None:
            return self._back_sub_df
        return self._raw_df

    @property
    def xdata(self):
        return self._xdata_of(
            self._raw_df,
            (pl.col('P_ANGLE') == self.p_angles[0])
            & (pl.col('A_DIFF_ANGLE') == self.a_diff_angles[0]),
        )

    def _xdata_of(self, df, filter):
        return np.array(
            df
            .filter(filter)
            .select(pl.col('WAVENUMBER'))
            .to_series()
        )

    def _ydata_of(self, df, filter):
        return np.array(
            df
            .filter(filter)
            .select(pl.col('COUNTS/SEC'))
            .to_series()
        )

    def _waterfall_plot(self, dfs, offset_factor, labels, colors):
        fig, axd = plt.subplot_mosaic(
            [self.a_diff_angles],
            sharex=True,
            sharey=True,
        )
        for li, (df, label, color) in enumerate(zip(dfs, labels, colors)):
            for ai, a in enumerate(self.a_diff_angles):
                for i, p in enumerate(self.p_angles):
                    label_ = None
                    if ai == 0 and i == 0:
                        label_ = label
                    xdata = self.xdata
                    if 'P_ANGLE' in df.columns:
                        filt = (
                            (pl.col('P_ANGLE') == p)
                            & (pl.col('A_DIFF_ANGLE') == a)
                        )
                    else:
                        filt = (pl.col('A_DIFF_ANGLE') == a)
                    ydata = self._ydata_of(
                        df,
                        filt,
                    )
                    norm_p = normalizer(
                        min(self.p_angles),
                        max(self.p_angles),
                    )(p)
                    offset = norm_p * offset_factor
                    axd[a].plot(xdata, ydata+offset, color=color, label=label_)
                    if li == 0:
                        axd[a].text(
                            xdata[-1],
                            (ydata+offset)[-1],
                            '${'+str(p)+r'}^\circ$',
                        )
        for a in self.a_diff_angles:
            axd[a].set_xlabel(r'$\nu$ (cm${}^{-1}$)')
            axd[a].set_ylabel(r'counts $\cdot$ s${}^{-1}$')
            axd[a].set_title('$a = '+str(a)+r'^\circ$')
        if not all([h is None for h in labels]):
            fig.legend()
        return fig, axd

    def waterfall(self, offset_factor=0):
        fig, axd = self._waterfall_plot(
            [
                self._df,
            ],
            offset_factor=offset_factor,
            labels=[
                None,
            ],
            colors=[
                'black',
            ],
        )
        return fig, axd

    def check_despike(self, offset_factor=0):
        fig, axd = self._waterfall_plot(
            [
                self._back_sub_df,
                self._despiked_df,
            ],
            offset_factor=offset_factor,
            labels=[
                'background subtracted',
                'despiked',
            ],
            colors=[
                'black',
                'red',
            ],
        )
        plt.show()

    def check_background(self, offset_factor=0):
        fig, axd = self._waterfall_plot(
            [
                self._raw_df,
                self._back_df,
                self._normalized_back_df,
                self._back_sub_df,
            ],
            offset_factor=offset_factor,
            labels=[
                'raw data',
                'raw background',
                'normalized background',
                'background subtracted',
            ],
            colors=[
                'black',
                'blue',
                'green',
                'red',
            ],
        )
        plt.show()

    def check_baseline(self, offset_factor=0):
        fig, axd = self._waterfall_plot(
            [
                self._despiked_df,
                self._baseline_df,
                self._base_sub_df,
            ],
            offset_factor=offset_factor,
            labels=[
                'despiked',
                'baseline',
                'baseline subtracted',
            ],
            colors=[
                'black',
                'blue',
                'red',
            ],
        )
        plt.show()

    def despike(self, ignore=[], threshold=12):

        self._despiked_df = pl.concat(
            [
                pl.LazyFrame(
                    {
                        'WAVENUMBER': self.xdata,
                        'COUNTS/SEC': despike(
                            self.xdata,
                            self._ydata_of(
                                self._df,
                                (pl.col('P_ANGLE') == p)
                                & (pl.col('A_DIFF_ANGLE') == a),
                            ),
                            ignore=ignore,
                        ),
                        'P_ANGLE': p,
                        'A_DIFF_ANGLE': a,
                    }
                )
                for p, a in zip(self._all_p_angles, self._all_a_diff_angles)
            ],
            how='vertical',
        ).collect()

    def _load_background(
        self,
        background_paths,
        background_a_diff_angles,
    ):
        self._back_df = pl.concat(
            [
                pl.scan_csv(
                    back_path,
                    comment_char='#',
                    encoding='utf8-lossy',
                    separator='\t',
                    has_header=False,
                    new_columns=['WAVENUMBER', 'COUNTS/SEC'],
                ).with_columns(
                    pl.lit(a).alias('A_DIFF_ANGLE'),
                ).sort(
                    pl.col('WAVENUMBER'),
                )
                for a, back_path in zip(
                    background_a_diff_angles.astype(int),
                    background_paths,
                )
            ],
            how='vertical',
        ).collect()

    def _subtract_background(
        self,
        background_reference_peak,
    ):
        ref = background_reference_peak
        normalized_back_qs = []
        for a in self.a_diff_angles:
            xdata = self._xdata_of(
                self._raw_df,
                (pl.col('P_ANGLE') == self.p_angles[0])
                & (pl.col('A_DIFF_ANGLE') == a),
            )
            ydata = self._ydata_of(
                self._raw_df,
                (pl.col('P_ANGLE') == self.p_angles[0])
                & (pl.col('A_DIFF_ANGLE') == a),
            )
            back_ydata = self._ydata_of(
                self._back_df,
                pl.col('A_DIFF_ANGLE') == a,
            )
            peaks, _ = find_peaks(ydata, distance=4)
            peaks_x = xdata[peaks]
            peak_ref = peaks[np.argmin(np.absolute(peaks_x-ref))]
            prominence = peak_prominences(ydata, [peak_ref])[0]
            back_peaks, _ = find_peaks(back_ydata, distance=4)
            back_peaks_x = xdata[back_peaks]
            back_peak_ref = back_peaks[
                np.argmin(
                    np.absolute(
                        back_peaks_x-ref,
                    ),
                )
            ]
            back_prominence = peak_prominences(back_ydata, [back_peak_ref])[0]
            normalized_back_ydata = back_ydata * prominence / back_prominence
            normalized_back_qs.append(
                self._back_df
                .lazy()
                .filter(pl.col('A_DIFF_ANGLE') == a)
                .select(
                    pl.lit(normalized_back_ydata)
                    .alias('COUNTS/SEC'),
                    pl.col('*').exclude('COUNTS/SEC'),
                )
            )
        normalized_back_q = pl.concat(normalized_back_qs, how='vertical')
        self._normalized_back_df = normalized_back_q.collect()
        self._back_sub_df = (
            self._raw_df
                .lazy()
                .join(
                    normalized_back_q,
                    on=['WAVENUMBER', 'A_DIFF_ANGLE'],
                    how='outer',
                    suffix='_background',
                ).select(
                    pl.col('WAVENUMBER'),
                    pl.col('P_ANGLE'),
                    pl.col('A_DIFF_ANGLE'),
                    pl.col('COUNTS/SEC') - pl.col('COUNTS/SEC_background')
                ).collect()
        )

    def subtract_background(
        self,
        background_paths,
        background_a_diff_angles,
        background_reference_peak,
    ):
        self._load_background(background_paths, background_a_diff_angles)
        self._subtract_background(background_reference_peak)

    def _gen_baseline(self, lam, p, niter=100, exclude=[], interactive=False):

        def print_and_return(x):
            print(x)
            return x

        rxclude = []
        for x1, x2 in exclude:
            r1 = np.argmin(abs(self.xdata-x1))
            r2 = np.argmin(abs(self.xdata-x2))
            rxclude.append((r1, r2))
        self._baseline_df = pl.concat(
            [
                pl.LazyFrame(
                    {
                        'WAVENUMBER': self.xdata,
                        'COUNTS/SEC': baseline(
                            self._ydata_of(
                                self._df,
                                (pl.col('P_ANGLE') == p_)
                                & (pl.col('A_DIFF_ANGLE') == a),
                            ),
                            lam,
                            p,
                            niter=niter,
                            exclude=rxclude,
                            xdata=self.xdata,
                            interactive=interactive,
                        ),
                        'P_ANGLE': p_,
                        'A_DIFF_ANGLE': a,
                    }
                )
                for p_, a in zip(self._all_p_angles, self._all_a_diff_angles)
            ],
            how='vertical',
        ).collect()

    def _subtract_baseline(self):
        self._base_sub_df = self._df \
            .lazy() \
            .join(
                self._baseline_df.lazy(),
                on=['WAVENUMBER', 'A_DIFF_ANGLE', 'P_ANGLE'],
                how='outer',
                suffix='_baseline',
            ).select(
                pl.col('WAVENUMBER'),
                pl.col('P_ANGLE'),
                pl.col('A_DIFF_ANGLE'),
                pl.col('COUNTS/SEC') - pl.col('COUNTS/SEC_baseline'),
            ).collect()

    def get_baseline(self, lam=None, p=None, niter=100,
                     exclude=[], interactive=False):
        if self._baseline_df is None:
            self._gen_baseline(
                lam,
                p,
                niter=niter,
                exclude=exclude,
                interactive=interactive,
            )
        return _Baseline(self._baseline_df)

    def subtract_baseline(self, lam, p, niter=100, exclude=[],
                          baseline=None, interactive=False):
        if baseline is None:
            self._gen_baseline(
                lam,
                p,
                niter=niter,
                exclude=exclude,
                interactive=interactive,
            )
        else:
            self._baseline_df = baseline._baseline_df
        self._subtract_baseline()

    def to_pickle(self, path):
        with open(path, 'wb') as fh:
            pickle.dump(self, fh)

    def get_modedata(self, center_frequency, left_bound, right_bound):
        pdatas = []
        adatas = []
        ydatas = []
        for a in self.a_diff_angles:
            df = self._df.filter(
                (pl.col('A_DIFF_ANGLE') == a)
                & (pl.col('WAVENUMBER') >= left_bound)
                & (pl.col('WAVENUMBER') < right_bound)
            ).pivot(
                index='P_ANGLE',
                columns='WAVENUMBER',
                values='COUNTS/SEC',
            ).select(
                pl.col('P_ANGLE'),
                pl.sum_horizontal(pl.col('*').exclude('P_ANGLE')),
            ).sort(by='P_ANGLE')
            pdata = np.array(df.select(pl.col('P_ANGLE')).to_series())
            adata = pdata + a
            ydata = np.array(df.select(pl.col('sum')).to_series())
            pdatas.append(pdata)
            adatas.append(adata)
            ydatas.append(ydata)
        _modedata = _ModeData(
            np.array(pdatas),
            np.array(adatas),
            np.array(ydatas),
        )
        return ModeData(center_frequency, left_bound, right_bound, _modedata)


class _Baseline:

    def __init__(self, df):
        self.df = df
