import pandas as pd

df = pd.read_csv('amtl.csv')
missing = df['num_amtl']
total = df['age']
print('invalid_neg', (missing<0).sum(), 'invalid_over', (missing>total).sum())
print('invalid any', ((missing<0)|(missing>total)).mean())
print('min missing', missing.min(), 'max missing', missing.max())
