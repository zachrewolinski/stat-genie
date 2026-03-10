import pandas as pd
import numpy as np
from pathlib import Path

df = pd.read_csv('reading.csv')
print('shape', df.shape)
print(df.head())
print(df.describe(include='all').transpose().head(25))

# Basic check for feature20 distribution
for col in ['feature4','feature5','feature6','feature7','feature20']:
    if col in df.columns:
        s = df[col]
        print('\n', col, 'min', s.min(), 'max', s.max(), 'mean', s.mean(), 'median', s.median())

# Check correlation between feature20 and words/time
if set(['feature7','feature5']).issubset(df.columns):
    # reading speed maybe words per minute: feature7 / (feature5/60000)
    speed_calc = df['feature7'] / (df['feature5'] / 60000)
    print('\ncalc speed words/min stats', speed_calc.describe())
    corr = np.corrcoef(speed_calc, df['feature20'])[0,1]
    print('corr feature20 vs calc speed', corr)

# Check relation between reader view and feature20 for dyslexia (feature17 or feature12)

