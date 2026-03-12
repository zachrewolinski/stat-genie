import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('reading.csv')
print(df.head())
print(df.describe(include='all').T[['count','mean','std','min','max']].head(10))
# compute wpm from feature7 words / feature5 (reading time minus scrolling)

df['wpm_f5'] = df['feature7'] * 60000 / df['feature5']

df['wpm_f4'] = df['feature7'] * 60000 / df['feature4']

for col in ['feature20','wpm_f5','wpm_f4']:
    print(col, df[col].describe())

print('corr feature20 vs wpm_f5', df['feature20'].corr(df['wpm_f5']))
print('corr feature20 vs wpm_f4', df['feature20'].corr(df['wpm_f4']))
