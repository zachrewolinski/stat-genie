import pandas as pd
import numpy as np
from statsmodels.stats.weightstats import ttest_ind

# Load data

df = pd.read_csv('reading.csv')

# Feature mapping from info.json:
# feature3 = reader view (1 on, 0 off)
# feature17 = dyslexia (1 yes, 0 no)
# feature20 = reading speed (words per minute; derived from words/time)

# Subset to participants with dyslexia and valid reading speed / reader view indicators
sub = df[(df['feature17'] == 1) & df['feature20'].notna() & df['feature3'].notna()]

on = sub[sub['feature3'] == 1]['feature20']
off = sub[sub['feature3'] == 0]['feature20']

# Summary stats
summary = {
    'n_on': int(on.shape[0]),
    'n_off': int(off.shape[0]),
    'mean_on': float(on.mean()),
    'mean_off': float(off.mean()),
    'median_on': float(on.median()),
    'median_off': float(off.median()),
}

# Welch's t-test
if len(on) > 1 and len(off) > 1:
    tstat, pval, dfree = ttest_ind(on, off, usevar='unequal')
else:
    tstat, pval, dfree = (np.nan, np.nan, np.nan)

# Cohen's d

def cohens_d(x, y):
    nx = len(x)
    ny = len(y)
    if nx < 2 or ny < 2:
        return np.nan
    vx = x.var(ddof=1)
    vy = y.var(ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    return (x.mean() - y.mean()) / np.sqrt(pooled)


d = cohens_d(on, off)

print('Reading speed (feature20) for dyslexic readers:')
print(f"n_on={summary['n_on']} n_off={summary['n_off']}")
print(f"mean_on={summary['mean_on']:.3f} mean_off={summary['mean_off']:.3f}")
print(f"median_on={summary['median_on']:.3f} median_off={summary['median_off']:.3f}")
print(f"Welch t-test: t={tstat:.3f} p={pval:.4f} df={dfree:.1f}")
print(f"Cohen's d={d:.3f}")
