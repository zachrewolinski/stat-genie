import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')

print('Head:')
print(df.head())

num_cols = df.select_dtypes(include=[np.number]).columns
for col in num_cols:
    print(f"\n{col} summary")
    print(df[col].describe())
    # check if close to integers
    frac = (df[col] - df[col].round()).abs().max()
    print('max fractional abs:', frac)
    print('n_unique:', df[col].nunique())
    # show small sample of unique sorted values
    vals = np.sort(df[col].unique())
    print('unique sample:', vals[:10], '... ', vals[-10:])

# show counts for categorical
for col in df.select_dtypes(exclude=[np.number]).columns:
    print(f"\n{col} unique values:")
    print(df[col].value_counts().head(10))

# check potential binomial: num_amtl vs genus?
print('\nCorrelation between numeric columns:')
print(df[num_cols].corr())

