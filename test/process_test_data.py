from pathlib import Path
import re

import numpy as np
import raman

DATA_DIR_0K = Path.cwd() / 'test_data' / 'pol_dep' / '0K'
OFFSET_FACTOR = 40

psweeps = {}
for d in [DATA_DIR_0K]:
    paths = []
    p_angles = []
    a_diff_angles = []
    for path in sorted(d.glob('*')):
        tempstr, channel, p = re.search(
            r'_([0-9]*)K_.*(XX|XY)_([0-9]*)deg',
            path.name,
        ).groups()
        paths.append(path)
        p_angles.append(float(p))
        if channel == 'XX':
            a_diff_angles.append(0)
        else:
            a_diff_angles.append(90)

    psweep = raman.PolarizationSweepData(np.array(p_angles), np.array(a_diff_angles), paths)
    psweeps[int(tempstr)] = psweep

fig, axd = psweeps[0].waterfall(OFFSET_FACTOR)

Path.mkdir(Path.cwd() / 'test_pkl', exist_ok=True)
psweeps[0].to_pickle(Path.cwd() / 'test_pkl' / 'data_0K.pkl')
