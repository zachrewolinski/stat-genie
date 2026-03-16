import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('reading.csv')

# examine correlation of feature20 with time and words
for time_col in ['feature4','feature5','feature6']:
    corr = df['feature20'].corr(df[time_col])
    print('corr feature20 with', time_col, corr)

# compare to ms per word and wpm
ms_per_word = df['feature4'] / df['feature7']
wpm = df['feature7'] / (df['feature4'] / 1000) * 60
print('corr feature20 with ms_per_word', df['feature20'].corr(ms_per_word))
print('corr feature20 with wpm', df['feature20'].corr(wpm))
print('feature20 mean', df['feature20'].mean(), 'ms_per_word mean', ms_per_word.mean(), 'wpm mean', wpm.mean())

# check ratios
sample = df[['feature4','feature7','feature20']].head(10)
print(sample)
print('ms_per_word head', (sample['feature4']/sample['feature7']).values)
print('wpm head', (sample['feature7']/(sample['feature4']/1000)*60).values)

# dyslexia counts
print('feature17 counts', df['feature17'].value_counts(dropna=False))
print('feature12 counts', df['feature12'].value_counts(dropna=False))

# check consistency between feature17 and feature12 (dyslexia severity?)
print('crosstab', pd.crosstab(df['feature17'], df['feature12']))
