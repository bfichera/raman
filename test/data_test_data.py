from pathlib import Path

import matplotlib.pyplot as plt

from raman import load_polarization_sweep

psweeps = {}
psweeps[0] = load_polarization_sweep(Path.cwd() / 'test_pkl' / 'data_0K.pkl')
psweeps[0].waterfall(40)
plt.show()
