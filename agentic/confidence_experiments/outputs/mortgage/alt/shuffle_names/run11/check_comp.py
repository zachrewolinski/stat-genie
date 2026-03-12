import pandas as pd

df = pd.read_csv('mortgage.csv')

# check complement
comp = (df['accept'] == 1 - df['deny']).mean()
print('accept == 1-deny proportion', comp)
print('accept + deny unique', (df['accept'] + df['deny']).value_counts())
