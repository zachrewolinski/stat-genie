import pandas as pd
import numpy as np

path = 'reading.csv'
df = pd.read_csv(path)

num_words = df['num_words']

candidates = ['adjusted_running_time','age','gender','running_time']
print('Correlation with num_words:')
for col in candidates:
    s = df[col]
    # Spearman for monotonic
    rho = s.corr(num_words, method='spearman')
    pear = s.corr(num_words, method='pearson')
    print(col, 'pearson', pear, 'spearman', rho)

# check correlations among time-like columns
print('\nInter-correlations:')
for i, c1 in enumerate(candidates):
    for c2 in candidates[i+1:]:
        print(c1, c2, 'pearson', df[c1].corr(df[c2]))

