import pandas as pd

df = pd.read_csv('mortgage.csv')
print(df[['feature11','feature14']].head())
print('feature11 mean', df['feature11'].mean())
print('feature14 mean', df['feature14'].mean())
print('sum counts', (df['feature11'] + df['feature14']).value_counts(dropna=False).to_dict())
print('correlation', df['feature11'].corr(df['feature14']))
print('unique11', sorted(df['feature11'].dropna().unique()))
print('unique14', sorted(df['feature14'].dropna().unique()))
print('n', len(df))
