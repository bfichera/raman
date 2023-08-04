from data import psweeps
import lmfit
from raman.tensor import RamanTensor
from raman.model import ModeModel
from raman.minimize import minimize_single, check_single
import numpy as np
from sympy.abc import a, b, c, d, e, f

modedata = psweeps[30].get_modedata(57.4, 55.749, 58.623)

modedata.plot()

t = np.array(
    [
        [a, d, e],
        [d, a, f],
        [e, f, c],
    ],
)
ramantensor = RamanTensor(t)
model = ModeModel(ramantensor)
params = model.guess(modedata)

models, result = minimize_single(
    [ramantensor],
    [modedata],
    shift=None,
    method='dual_annealing',
    bound=10,
    no_local_search=True,
)
print(lmfit.fit_report(result))
check_single(models, [modedata], result.params)
