import numpy as np
import lmfit
import matplotlib.pyplot as plt

from .model import ModeModel


def minimize_single(ramantensors, modedatas, params=None, shift=None,
                    bound=None, **kwargs):

    models = []
    prefixes = []
    if params is not None:
        submitted_params = params.copy()
    else:
        submitted_params = None
    params = lmfit.Parameters()
    for i, (ramantensor, modedata) in enumerate(zip(ramantensors, modedatas)):
        prefixes.append(f't{int(modedata.center_frequency)}_')
        m = ModeModel(ramantensor, prefix=prefixes[i], a_diff_angles=modedata.a_diff_angles)
        pars = m.guess(modedata)
        if bound is not None:
            for name in pars:
                pars[name].set(min=-bound, max=bound)
        params.add_many(*tuple(pars.values()))
        models.append(m)
    if shift is None:
        params.add('shift', value=0, min=-180, max=180)
    else:
        params.add('shift', value=shift, min=-180, max=180, vary=False)
    for name in params:
        if 'shift' in name and name != 'shift':
            params[name].set(expr='shift')

    if submitted_params is not None:
        params = submitted_params.copy()

    def resid(params):
        result = np.array(
            [
                (
                    m.eval(
                        params,
                        p_angle=d.flattened_pdata,
                        a_angle=d.flattened_adata,
                    )
                    - d.flattened_ydata
                )
                for i, (m, d) in enumerate(zip(models, modedatas))
            ],
        ).flatten()
        return result

    return models, lmfit.minimize(resid, params, **kwargs)


def check_single(models, modedatas, params):

    fig, axd = plt.subplot_mosaic([np.arange(len(models))])
    for i, (m, d) in enumerate(zip(models, modedatas)):
        axd[i].set_title(d.center_frequency)
        for a in d.a_diff_angles:
            pdata = d.pdata_of(a)
            ydata = d.ydata_of(a)
            axd[i].plot(
                pdata,
                ydata,
                label='$a='+str(a)+r'^\circ$ (data)',
            )
            model_p = np.linspace(min(pdata), max(pdata), 5000)
            model_y = m.eval(params, p_angle=model_p, a_angle=model_p+a)
            axd[i].plot(
                model_p,
                model_y,
                label='$a='+str(a)+r'^\circ$ (fit)',
            )
    plt.legend()
    plt.show()












