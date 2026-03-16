import pandas as pd
import numpy as np
from scipy import stats


df = pd.read_csv('reading.csv')

reader_view_col = 'language'

# candidate dyslexia indicators: device (1/2), dyslexia_bin (1), correct_rate (1)
# compute effects for running_time
candidates = {
    'device_1or2': df['device'].isin([1.0,2.0]),
    'dyslexia_bin_1': df['dyslexia_bin']==1.0,
    'correct_rate_1': df['correct_rate']==1.0,
}

for name, mask in candidates.items():
    sub = df.loc[mask, [reader_view_col,'running_time']].dropna()
    rv0 = sub.loc[sub[reader_view_col]==0, 'running_time']
    rv1 = sub.loc[sub[reader_view_col]==1, 'running_time']
    if len(rv0) < 2 or len(rv1) < 2:
        print(name, 'insufficient data')
        continue
    t_stat, p_val = stats.ttest_ind(rv1, rv0, equal_var=False)
    print('\n', name, 'n', len(sub), 'p', p_val)
    print('means', rv0.mean(), rv1.mean(), 'medians', rv0.median(), rv1.median())
