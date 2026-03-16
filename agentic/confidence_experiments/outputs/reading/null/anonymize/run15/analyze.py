import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')
print(df.head())
print(df.describe(include='all').T.head(25))

# check if feature20 equals words per minute from time feature5 or feature4
for time_col in ['feature4','feature5']:
    wpm = df['feature7'] / (df[time_col]/60000.0)
    # compute correlation with feature20
    corr = np.corrcoef(wpm, df['feature20'])[0,1]
    # compute median absolute relative difference
    rel_diff = np.median(np.abs(wpm - df['feature20']) / np.maximum(np.abs(wpm), 1e-9))
    print(time_col, 'corr with feature20', corr, 'median rel diff', rel_diff)

# Also check if feature20 equals ms per word?
for time_col in ['feature4','feature5']:
    ms_per_word = df[time_col] / df['feature7']
    corr = np.corrcoef(ms_per_word, df['feature20'])[0,1]
    rel_diff = np.median(np.abs(ms_per_word - df['feature20']) / np.maximum(np.abs(ms_per_word), 1e-9))
    print(time_col, 'ms_per_word corr with feature20', corr, 'median rel diff', rel_diff)

print('feature20 summary', df['feature20'].describe())
