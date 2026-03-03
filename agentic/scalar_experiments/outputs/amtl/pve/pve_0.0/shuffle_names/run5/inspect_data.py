import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')

num_cols = df.select_dtypes(include=[np.number]).columns
print('numeric cols', num_cols.tolist())

for col in num_cols:
    frac_int = np.mean(np.isclose(df[col], np.round(df[col])))
    print(col, 'frac_int', round(float(frac_int), 3), 'min', df[col].min(), 'max', df[col].max(), 'unique', df[col].nunique())

print('\nrows per prob_male (specimen id?)')
counts = df['prob_male'].value_counts()
print(counts.describe())
print('top counts')
print(counts.head(10))

ct = df.pivot_table(index='prob_male', columns='sockets', values='genus', aggfunc='size')
print('\nrows per specimen by sockets (first 5)')
print(ct.head())
print('any missing classes?', ct.isna().any(axis=1).mean())

print('\ncorrelation')
print(df[num_cols].corr())
