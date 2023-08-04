from pathlib import Path

import numpy as np

OUTPUT_DIR = Path.cwd() / 'test_data' / 'pol_dep' / '0K'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

pdata = np.arange(0, 360, 10)
a_diff_angles = [0, 90]

for a_diff_angle in a_diff_angles:
    for p_angle in pdata:
        if a_diff_angle == 0:
            channel = 'XX'
        elif a_diff_angle == 90:
            channel = 'XY'
        a_angle = p_angle+a_diff_angle
        a = 3.21
        d = 1.45
        pi = np.pi
        sin = np.sin
        cos = np.cos
        filename = f'AngleDep_0K_s1_20230804_{channel}_{p_angle}deg.txt'
        path = OUTPUT_DIR / filename
        xdata = np.linspace(25, 500, 5000)
        ydata = np.full(
            xdata.shape,
            fill_value=2*a**2*sin(pi*a_angle/180)**2*sin(pi*p_angle/180)**2 - a**2*sin(pi*a_angle/180)**2 + a**2*sin(pi*a_angle/90)*sin(pi*p_angle/90)/2 - a**2*sin(pi*p_angle/180)**2 + a**2 + 2*a*d*sin(pi*a_angle/180)*cos(pi*a_angle/180) + 2*a*d*sin(pi*p_angle/180)*cos(pi*p_angle/180) - 2*d**2*sin(pi*a_angle/180)**2*sin(pi*p_angle/180)**2 + d**2*sin(pi*a_angle/180)**2 + d**2*sin(pi*a_angle/90)*sin(pi*p_angle/90)/2 + d**2*sin(pi*p_angle/180)**2,
        )
        lines = []
        for x, y in zip(xdata, ydata):
            lines.append(f'{x}\t{y}\n')
        with open(path, 'w') as fh:
            fh.writelines(lines)
