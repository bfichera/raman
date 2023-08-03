from pathlib import Path

import matplotlib.pyplot as plt

from raman import load_polarization_sweep

psweeps = {}
psweeps[30] = load_polarization_sweep(Path.cwd() / 'pkl' / 'data_30K.pkl')
psweeps[180] = load_polarization_sweep(Path.cwd() / 'pkl' / 'data_180K.pkl')
psweeps[30].waterfall(40)
plt.show()
