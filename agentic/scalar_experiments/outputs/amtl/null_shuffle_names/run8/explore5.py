import pandas as pd

df = pd.read_csv('amtl.csv')
counts = df['prob_male'].value_counts()
print('specimen count min', counts.min(), 'max', counts.max())
print(counts.head())
