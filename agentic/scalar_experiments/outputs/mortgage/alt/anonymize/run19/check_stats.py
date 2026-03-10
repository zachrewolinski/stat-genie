import pandas as pd

df = pd.read_csv('mortgage.csv')
print(df[['feature2','feature14','feature11']].head())
print('feature14 counts:', df['feature14'].value_counts(dropna=False).to_dict())
print('feature11 counts:', df['feature11'].value_counts(dropna=False).to_dict())
print('mean accepted:', df['feature14'].mean())
print('mean denied:', df['feature11'].mean())
print('mean accepted by female:', df.groupby('feature2')['feature14'].mean().to_dict())
print('mean denied by female:', df.groupby('feature2')['feature11'].mean().to_dict())
