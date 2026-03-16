import pandas as pd
import numpy as np

path = 'amtl.csv'

df = pd.read_csv(path)
print(df.head())
print('\nColumns:', df.columns.tolist())
print('\nDtypes:', df.dtypes)
print('\nDescribe numeric:')
print(df.describe())

for col in df.columns:
    if df[col].dtype == 'object':
        print('\n', col, 'unique:', df[col].nunique())
        print('samples:', df[col].unique()[:5])

for col in df.columns:
    if np.issubdtype(df[col].dtype, np.number):
        is_int = np.allclose(df[col].dropna(), np.round(df[col].dropna()))
        print(f"\n{col} min={df[col].min():.3f} max={df[col].max():.3f} mean={df[col].mean():.3f} int_like={is_int}")

spec_col = 'prob_male'
print('\nSpecimen rows per id summary:')
print(df[spec_col].value_counts().describe())

for col in df.columns:
    if np.issubdtype(df[col].dtype, np.number):
        neg = (df[col] < 0).sum()
        print(col, 'negative count', neg)
