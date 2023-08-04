from data_test_data import psweeps
import lmfit
from raman.tensor import RamanTensor
from raman.model import ModeModel
from raman.minimize import minimize_single, check_single
import numpy as np
from sympy.abc import a, b, c, d, e, f

modedata = psweeps[0].get_modedata(57.4, 55.749, 58.623)

modedata.plot()

t = np.array(
    [
        [a, d, 0],
        [d, a, 0],
        [0, 0, c],
    ],
)
ramantensor = RamanTensor(t)
model = ModeModel(ramantensor)
params = model.guess(modedata)

models, result = minimize_single(
    [ramantensor],
    [modedata],
    shift=None,
    method='leastsq',
)
print(lmfit.fit_report(result))
check_single(models, [modedata], result.params)
