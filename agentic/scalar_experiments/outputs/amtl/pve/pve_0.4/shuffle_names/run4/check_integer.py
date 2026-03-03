import pandas as pd
import numpy as np

# Load data

df = pd.read_csv('amtl.csv')

# check closeness to integer
for col in ['genus','age','num_amtl']:
    vals = df[col].values
    diffs = np.abs(vals - np.round(vals))
    print(col, 'max diff to integer', diffs.max(), 'mean diff', diffs.mean())
    print('  pct within 1e-6', (diffs < 1e-6).mean())
    print('  pct within 1e-2', (diffs < 1e-2).mean())
