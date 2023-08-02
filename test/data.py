from pathlib import Path
import re

import raman

DATA_DIR_30K = Path.cwd() / 'data' / 'pol_dep' / '30K'
DATA_DIR_180K = Path.cwd() / 'data' / 'pol_dep' / '180K'
BACKGROUND_XX_PATH = Path.cwd() / 'data' / 'background' / 'CaMn2Bi2_XX_Background_600sx2_100x.txt'
BACKGROUND_XY_PATH = Path.cwd() / 'data' / 'background' / 'CaMn2Bi2_XY_Background_600sx2_100x.txt'
OFFSET_FACTOR = 40

psweeps = {}
for d in [DATA_DIR_30K, DATA_DIR_180K]:
    paths = []
    p_angles = []
    a_angles = []
    for path in sorted(d.glob('*')):
        tempstr, channel, p = re.search(r'_([0-9]*)K_.*(XX|XY)_([0-9]*)deg', path.name).groups()
        paths.append(path)
        p_angles.append(float(p))
        if channel == 'XX':
            a_angles.append(0)
        else:
            a_angles.append(45)

    back_paths = [BACKGROUND_XX_PATH, BACKGROUND_XY_PATH]
    back_a_angles = [0, 45]

    psweep = raman.PolarizationSweepData(p_angles, a_angles, paths)
    psweep.subtract_background(back_paths, back_a_angles, 60)
    psweep.check_background(OFFSET_FACTOR)
    psweep.despike(
        ignore=[
            (70, 90),
            (152, 164),
            (90, 95),
            (54, 60),
        ],
    )
    psweep.check_despike(OFFSET_FACTOR)
    psweeps[int(tempstr)] = psweep

psweeps[30].subtract_baseline(1e3, 0.01, 100, exclude=[(63.6, 120), (175, 500)], interactive=False)
psweeps[30].check_baseline(OFFSET_FACTOR)
psweeps[180].subtract_baseline(1e3, 0.01, 100, exclude=[(63.6, 120), (175, 500)], interactive=False)
psweeps[180].check_baseline(OFFSET_FACTOR)

psweeps[30].waterfall(OFFSET_FACTOR)
psweeps[180].waterfall(OFFSET_FACTOR)
