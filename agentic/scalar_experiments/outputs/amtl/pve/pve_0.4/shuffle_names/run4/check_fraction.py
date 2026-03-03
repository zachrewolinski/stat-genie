import pandas as pd
import numpy as np

# Load data

df = pd.read_csv('amtl.csv')

for col in ['genus','num_amtl']:
    frac = np.modf(df[col].values)[0]
    # round to 3 decimals
    uniq = np.unique(np.round(frac,3))
    print(col, 'unique fractional (rounded) count', len(uniq))
    print('sample', uniq[:20])
