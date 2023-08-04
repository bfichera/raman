import numpy as np
import lmfit
import matplotlib.pyplot as plt

from .model import ModeModel


def minimize_single(ramantensors, modedatas, shift=None,
                    rel_scale=None, bound=None, **kwargs):

    models = []
    prefixes = []
    params = lmfit.Parameters()
    for i, (ramantensor, modedata) in enumerate(zip(ramantensors, modedatas)):
        prefixes.append(f't{int(modedata.center_frequency)}_')
        m = ModeModel(ramantensor, prefix=prefixes[i])
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
    if rel_scale is None:
        params.add('rel_scale', value=1, min=0, max=np.inf)
    else:
        params.add('rel_scale', value=rel_scale, min=0, max=np.inf, vary=False)

    def resid(params):
        shift = params['shift'].value
        result = np.array(
            [
                (
                    m.eval(
                        params,
                        p_angle=d.flattened_pdata-shift,
                        a_angle=d.flattened_adata-shift,
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












