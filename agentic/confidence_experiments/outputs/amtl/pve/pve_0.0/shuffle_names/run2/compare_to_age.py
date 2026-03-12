import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')

for col in ['genus','pop','num_amtl','stdev_age']:
    if np.issubdtype(df[col].dtype, np.number):
        frac_le = (df[col] <= df['age']).mean()
        frac_le = float(frac_le)
        print(col, 'fraction <= age', frac_le, 'min', df[col].min(), 'max', df[col].max())

