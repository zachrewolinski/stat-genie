import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')

# Identify possible count columns: numeric columns with near-integer values
num_cols = df.select_dtypes(include=[np.number]).columns
print('numeric cols', num_cols.tolist())

for col in num_cols:
    # fraction of values close to integer
    frac_int = np.mean(np.isclose(df[col], np.round(df[col])))
    print(col, 'frac_int', frac_int, 'min', df[col].min(), 'max', df[col].max(), 'unique', df[col].nunique())

# Check group sizes per specimen id (prob_male column)
print('\nrows per prob_male (specimen id?)')
counts = df['prob_male'].value_counts()
print(counts.describe())
print('top counts')
print(counts.head(10))

# Check if each specimen has 3 tooth classes
ct = df.pivot_table(index='prob_male', columns='sockets', values='genus', aggfunc='size')
print('\nrows per specimen by sockets (first 5)')
print(ct.head())
print('any missing classes?', ct.isna().any(axis=1).mean())

# Correlation between numeric columns
print('\ncorrelation')
print(df[num_cols].corr())
