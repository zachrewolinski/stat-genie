import pandas as pd
import numpy as np

_df = pd.read_csv('amtl.csv')

for col in ['genus','num_amtl','pop']:
    diffs = np.abs(_df[col] - _df[col].round())
    print(col, 'mean abs diff to nearest int', diffs.mean(), 'median', diffs.median())
    print(col, 'frac within 0.1 of int', (diffs<0.1).mean())

