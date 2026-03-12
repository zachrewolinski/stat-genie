import pandas as pd
import numpy as np
from scipy.special import expit

# Load data

df = pd.read_csv('amtl.csv')

# treat genus as logit of proportion
p = expit(df['genus'])
df['p'] = p

print('p summary', p.min(), p.max(), p.mean(), p.std())
print('p by tooth_class', df.groupby('tooth_class')['p'].mean())

# check if p relates to sockets counts (age)
print('corr p age', df['p'].corr(df['age']))

# check if p relates to actual missing count? maybe we can approximate missing count = p * age
print('p*age summary', (p*df['age']).describe())

