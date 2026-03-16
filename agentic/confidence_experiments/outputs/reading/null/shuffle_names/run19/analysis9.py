import pandas as pd
import numpy as np
from scipy import stats


df = pd.read_csv('reading.csv')

# Candidate speed definitions
candidates = {
    'running_time_raw': df['running_time'],
    'speed_from_retake_adj_ms': df['retake_trial'] / (df['adjusted_running_time'] / 1000.0 / 60.0),
    'speed_from_retake_age_ms': df['retake_trial'] / (df['age'] / 1000.0 / 60.0),
    'speed_from_retake_gender_ms': df['retake_trial'] / (df['gender'] / 1000.0 / 60.0),
}

# dyslexia status in device; reader_view in language
for name, speed in candidates.items():
    sub = pd.DataFrame({'speed': speed, 'reader_view': df['language'], 'device': df['device']})
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
    sub = sub[sub['device'] > 0]
    rv_on = sub[sub['reader_view'] == 1]['speed']
    rv_off = sub[sub['reader_view'] == 0]['speed']
    if len(rv_on) < 2 or len(rv_off) < 2:
        continue
    tstat, pval = stats.ttest_ind(rv_on, rv_off, equal_var=False)
    # Cohen d
    n1, n2 = len(rv_on), len(rv_off)
    s1, s2 = rv_on.var(ddof=1), rv_off.var(ddof=1)
    pooled = ((n1-1)*s1 + (n2-1)*s2)/(n1+n2-2)
    d = (rv_on.mean() - rv_off.mean())/np.sqrt(pooled)
    print('\n', name)
    print('n_on', len(rv_on), 'n_off', len(rv_off))
    print('means', rv_on.mean(), rv_off.mean())
    print('p', pval, 'd', d)

