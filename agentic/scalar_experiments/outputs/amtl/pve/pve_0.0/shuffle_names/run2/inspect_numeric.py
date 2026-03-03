import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')

for col in ['genus','age','pop','num_amtl','stdev_age']:
    vals = df[col].values
    frac_int = np.mean(np.isclose(vals, np.round(vals), atol=1e-6))
    print(col, 'frac_int', frac_int, 'unique', df[col].nunique())

# Check if any numeric columns are bounded between 0 and 1 (sex probability)
for col in ['genus','age','pop','num_amtl','stdev_age']:
    print(col, 'min', df[col].min(), 'max', df[col].max())

