import pandas as pd

df = pd.read_csv('mortgage.csv')
ct = pd.crosstab(df['feature2'], df['feature14'])
print(ct)
print('row sums', ct.sum(axis=1).to_dict())
print('col sums', ct.sum(axis=0).to_dict())
print('accept_rate_male', ct.loc[0,1]/ct.loc[0].sum())
print('accept_rate_female', ct.loc[1,1]/ct.loc[1].sum())
