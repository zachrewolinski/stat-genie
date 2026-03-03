import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
logit = df['genus'].to_numpy()
p = 1/(1+np.exp(-logit))

a = df['age'].to_numpy()
# closeness of p to k/a
# compute for each row min abs difference between p and any k/a
max_a = a.max()
# precompute possible proportions for each a
props = {val: (np.arange(val+1) / val) for val in np.unique(a)}
min_diffs = np.array([np.min(np.abs(pi - props[ai])) for pi, ai in zip(p, a)])
print('min diff proportion mean', min_diffs.mean())
print('min diff proportion median', np.median(min_diffs))
print('min diff proportion <=0.01', (min_diffs<=0.01).mean())
print('min diff proportion <=0.05', (min_diffs<=0.05).mean())
# also check if p*age near integer
est_missing = p * a
frac = np.abs(est_missing - np.round(est_missing))
print('mean abs frac', frac.mean())
print('median abs frac', np.median(frac))
print('within 0.1 of int', (frac<0.1).mean())
