import inspect

import numpy as np
from lmfit.model import Model
from lmfit.models import update_param_vals

class ModeModel(Model):

    def __init__(self, ramantensor, a_diff_angles=[], independent_vars=['p_angle', 'a_angle'],
                 prefix='', nan_policy='raise', **kwargs):
        kwargs.update(
            {
                'prefix': prefix,
                'nan_policy': nan_policy,
                'independent_vars': independent_vars,
            },
        )
        model_func = ramantensor.get_model_func()
        arg_names = [p.name for p in inspect.signature(model_func).parameters.values()]
        background_names = ['back_{a_}' for a_ in a_diff_angles]
        super().__init__(model_func, **kwargs)
        self._set_paramhints_prefix()

    def guess(self, modedata, **kwargs): 
        ydata = modedata.sample_ydata()
        pars = self.make_params()
        for name in pars:
            pars[name].set(value=np.sqrt(np.mean(ydata)))
        return update_param_vals(pars, self.prefix, **kwargs)
