import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('reading.csv')
subset = df[df['feature12'].isin([1,2])].dropna(subset=['feature3','feature5','feature7'])
subset['wpm_reading'] = subset['feature7'] / (subset['feature5']/60000.0)
rv_on = subset[subset['feature3']==1]['wpm_reading']
rv_off = subset[subset['feature3']==0]['wpm_reading']

t = stats.ttest_ind(rv_on, rv_off, equal_var=False)
mu = stats.mannwhitneyu(rv_on, rv_off, alternative='two-sided')
print('n on', len(rv_on), 'n off', len(rv_off))
print('means', rv_on.mean(), rv_off.mean())
print('medians', rv_on.median(), rv_off.median())
print('t', t)
print('mw', mu)
