import pandas as pd
import numpy as np
from scipy import stats


df = pd.read_csv('affairs.csv')

num_cols = ['education','age','occupation','children','rating','yearsmarried','rownames','affairs']

for col in ['education','age']:
    print('\nCandidate', col)
    for other in ['occupation','children','rating','yearsmarried','rownames','affairs']:
        r, p = stats.spearmanr(df[col], df[other])
        print(f'  spearman with {other}: r={r:.3f}, p={p:.3g}')
