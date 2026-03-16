import pandas as pd

df = pd.read_csv('mortgage.csv')
print(pd.crosstab(df['accept'], df['deny'], dropna=False))

# check if accept/deny are complements
print('accept+deny unique', (df['accept'] + df['deny']).value_counts(dropna=False).sort_index())

# list rows where accept+deny !=1
bad = df[df['accept'] + df['deny'] != 1]
print('rows where accept+deny !=1', bad.shape)
print(bad[['accept','deny']].head())

