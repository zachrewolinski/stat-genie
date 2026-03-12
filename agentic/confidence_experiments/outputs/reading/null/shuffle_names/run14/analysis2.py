import pandas as pd
import numpy as np
from scipy import stats


df = pd.read_csv('reading.csv')

participant_id = 'speed'
reader_view = 'language'
dyslexia_status = 'device'

# Dyslexic subset
subset = df[df[dyslexia_status].isin([1.0,2.0])].copy()
subset = subset[subset[reader_view].isin([0,1])]

# candidate speed measures
subset['speed_running_time'] = subset['running_time']
subset['speed_from_age'] = subset['retake_trial'] / subset['age'] * 60000
subset['speed_from_adjusted'] = subset['retake_trial'] / subset['adjusted_running_time'] * 60000

measures = ['speed_running_time','speed_from_age','speed_from_adjusted']

for m in measures:
    data = subset[[reader_view, m]].dropna()
    rv0 = data.loc[data[reader_view]==0, m]
    rv1 = data.loc[data[reader_view]==1, m]
    u_stat, p_mw = stats.mannwhitneyu(rv1, rv0, alternative='two-sided')
    mean0, mean1 = rv0.mean(), rv1.mean()
    med0, med1 = rv0.median(), rv1.median()
    print('\n', m)
    print('n0', len(rv0), 'n1', len(rv1))
    print('mean0', mean0, 'mean1', mean1)
    print('median0', med0, 'median1', med1)
    print('p', p_mw)
