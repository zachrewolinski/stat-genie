import pandas as pd
import numpy as np
from statsmodels.stats.weightstats import ttest_ind
import statsmodels.api as sm

# Load data
_df = pd.read_csv('reading.csv')

# Focus on dyslexic participants
_df = _df[_df['dyslexia_bin'] == 1].copy()
_df = _df[['speed', 'reader_view', 'num_words', 'Flesch_Kincaid', 'device', 'page_id', 'uuid']].dropna(subset=['speed', 'reader_view'])

# Basic group stats
_group = _df.groupby('reader_view')['speed']
summary = _group.agg(['count', 'mean', 'median', 'std']).rename(index={0: 'No Reader View', 1: 'Reader View'})

# Welch's t-test on raw speed
rv = _df[_df['reader_view'] == 1]['speed']
no = _df[_df['reader_view'] == 0]['speed']

t_stat, p_val, dfree = ttest_ind(rv, no, usevar='unequal')

# Robust check: log-transform due to heavy skew
_df['log_speed'] = np.log1p(_df['speed'])
rv_log = _df[_df['reader_view'] == 1]['log_speed']
no_log = _df[_df['reader_view'] == 0]['log_speed']

t_stat_log, p_val_log, dfree_log = ttest_ind(rv_log, no_log, usevar='unequal')

# Regression with controls (log speed)
# Use simple controls to adjust for page/text difficulty and device
X = pd.get_dummies(_df[['reader_view', 'num_words', 'Flesch_Kincaid', 'device', 'page_id']], drop_first=True)
X = sm.add_constant(X)
model = sm.OLS(_df['log_speed'], X).fit(cov_type='HC3')

# Output results for inspection
print('Dyslexic participants only')
print(summary)
print('\nWelch t-test (raw speed): t=%.3f p=%.4f n_RV=%d n_No=%d' % (t_stat, p_val, len(rv), len(no)))
print('Welch t-test (log speed): t=%.3f p=%.4f' % (t_stat_log, p_val_log))

coef = model.params.get('reader_view', np.nan)
se = model.bse.get('reader_view', np.nan)
pv = model.pvalues.get('reader_view', np.nan)
print('\nRegression (log speed) reader_view coef=%.4f se=%.4f p=%.4f' % (coef, se, pv))

# Translate log coefficient into % change for readability
if pd.notna(coef):
    pct = (np.expm1(coef)) * 100
    print(f'Approx % change in speed with reader_view: {pct:.2f}%')
