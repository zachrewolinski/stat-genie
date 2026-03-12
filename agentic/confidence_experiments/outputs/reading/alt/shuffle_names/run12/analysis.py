import pandas as pd
import numpy as np

path = 'reading.csv'
df = pd.read_csv(path)
print('shape', df.shape)
print('columns', df.columns.tolist())
print(df.head(3))
print(df.dtypes)
print('\nMissing counts top:')
print(df.isna().sum().sort_values(ascending=False).head(10))

# Show unique values for small-cardinality columns
for col in df.columns:
    nunique = df[col].nunique(dropna=False)
    if nunique <= 10:
        print('\n', col, 'nunique', nunique)
        print(df[col].value_counts(dropna=False))

# Describe numeric columns
print('\nNumeric describe:')
print(df.describe(include=[np.number]).T)

