import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
for col in ['genus','num_amtl','pop','stdev_age']:
    frac = (df[col] - np.floor(df[col])).round(3)
    print(col, frac.value_counts().head(10))
