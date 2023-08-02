from raman import load_polarization_sweep

psweeps = {}
psweeps[30] = load_polarization_sweep(Path.cwd() / 'data_30K.pkl')
psweeps[180] = load_polarization_sweep(Path.cwd() / 'data_180K.pkl')
