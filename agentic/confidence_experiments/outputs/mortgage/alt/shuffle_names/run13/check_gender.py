import pandas as pd

df = pd.read_csv('mortgage.csv')
print('denied_PMI mean', df['denied_PMI'].mean())
print('denied_PMI value counts', df['denied_PMI'].value_counts())
