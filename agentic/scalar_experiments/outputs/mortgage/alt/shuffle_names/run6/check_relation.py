import pandas as pd

df = pd.read_csv('mortgage.csv')

print('deny + self_employed value_counts')
print((df['deny'] + df['self_employed']).value_counts())
print('deny == 1 - self_employed', (df['deny'] == (1 - df['self_employed'])).mean())

