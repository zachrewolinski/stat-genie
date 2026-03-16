import pandas as pd
import numpy as np

cols = ['genus','age','pop','num_amtl','stdev_age']

df = pd.read_csv('amtl.csv')

for missing_col in cols:
    for total_col in cols:
        if missing_col == total_col:
            continue
        missing = df[missing_col]
        total = df[total_col]
        # only consider totals with positive values
        invalid = ((missing < 0) | (missing > total)).mean()
        # require total positive typical
        if total.min() <= 0:
            pass
        print(missing_col, 'over', total_col, 'invalid', round(invalid,3))
