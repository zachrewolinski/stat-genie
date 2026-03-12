import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
missing = df['genus']
total = df['age']

invalid_neg = (missing < 0).sum()
invalid_over = (missing > total).sum()
print('invalid_neg', invalid_neg, 'invalid_over', invalid_over, 'total rows', len(df))
print('min missing', missing.min(), 'max missing', missing.max())
print('min total', total.min(), 'max total', total.max())

print('invalid any', ((missing<0)|(missing>total)).mean())
