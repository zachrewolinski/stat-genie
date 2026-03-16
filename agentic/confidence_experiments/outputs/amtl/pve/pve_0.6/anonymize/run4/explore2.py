import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
# share how many values within 0..1
within = df['feature3'].between(0,1).mean()
print('feature3 within 0-1:', within)
print('min', df['feature3'].min(), 'max', df['feature3'].max())
# proportion rate if treat feature3 as count
rate = df['feature3'] / df['feature4']
print('rate min', rate.min(), 'max', rate.max(), 'mean', rate.mean())
print('rate within 0-1', rate.between(0,1).mean())
# check if feature3 roughly in 0-14 maybe scaled? find correlation with feature4
print('corr feature3 feature4', df['feature3'].corr(df['feature4']))
# check feature3 * feature4 distribution
prod = df['feature3'] * df['feature4']
print('feature3*feature4 min', prod.min(), 'max', prod.max(), 'mean', prod.mean())
