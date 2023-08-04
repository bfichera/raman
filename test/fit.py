from data import psweeps
import lmfit
from raman.tensor import RamanTensor, real_and_imag
from raman.model import ModeModel
from raman.minimize import minimize_single, check_single
import numpy as np
from sympy.abc import a, b, c, d, e, f

modedata = psweeps[30].get_modedata(
    57.4,
    55.749,
    58.623,
)
# modedata = psweeps[30].get_modedata(
#     464.93,
#     450.00,
#     476.45,
# )
# modedata = psweeps[30].get_modedata(
#     78.1,
#     76.1,
#     80.1,
# )
# modedata = psweeps[30].get_modedata(
#     92.8,
#     90.8,
#     94.8,
# )

modedata.plot()

t = np.array(
    [
        [a, d, e],
        [d, b, f],
        [e, f, c],
    ],
)
t_real, t_imag = real_and_imag(t)
ramantensor = RamanTensor(
    t_real,
    t_imag,
)
model = ModeModel(ramantensor)
params = model.guess(modedata)

models, result = minimize_single(
    [ramantensor],
    [modedata],
    shift=None,
    bound=1,
    method='powell',
)
print(lmfit.fit_report(result))
check_single(models, [modedata], result.params)
