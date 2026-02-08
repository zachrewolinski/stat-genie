import pandas as pd
import numpy as np

path = 'amtl.csv'
df = pd.read_csv(path)
print(df.head())
print('\nDtypes:')
print(df.dtypes)
print('\nSummary numeric:')
print(df.describe(include=[np.number]).T)

# integer-likeness
for col in df.select_dtypes(include=[np.number]).columns:
    vals = df[col].dropna()
    frac = np.mean(np.isclose(vals, np.round(vals)))
    print(col, 'integer_like_frac', frac)

# unique counts for categorical
for col in df.select_dtypes(exclude=[np.number]).columns:
    print(col, 'unique', df[col].nunique(), 'sample', df[col].unique()[:5])
