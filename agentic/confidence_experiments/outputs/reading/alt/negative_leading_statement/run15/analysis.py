import pandas as pd
import numpy as np
from scipy import stats

# Load data

df = pd.read_csv('reading.csv')

# Subset to dyslexia participants
# use dyslexia_bin == 1 (has dyslexia)

dys = df[df['dyslexia_bin'] == 1].copy()

# Basic counts
n_total = len(df)
n_dys = len(dys)

# Group by reader_view

groups = dys.groupby('reader_view')
counts = groups['speed'].count()

# Descriptive stats

desc = groups['speed'].agg(['mean','median','std','min','max'])

# Also compute log-speed stats (avoid log(0))
# speed min >0? check
min_speed = dys['speed'].min()
# log1p to be safe

dys['log_speed'] = np.log1p(dys['speed'])
log_desc = dys.groupby('reader_view')['log_speed'].agg(['mean','median','std'])

# Tests
# Mann-Whitney U (nonparam)

g0 = dys[dys['reader_view'] == 0]['speed']
g1 = dys[dys['reader_view'] == 1]['speed']

mw = stats.mannwhitneyu(g1, g0, alternative='two-sided')

# t-test on log_speed

tt = stats.ttest_ind(dys[dys['reader_view']==1]['log_speed'], dys[dys['reader_view']==0]['log_speed'], equal_var=False, nan_policy='omit')

# Effect size: Cliff's delta for speed

def cliffs_delta(x, y):
    # compute via ranks
    # from https://en.wikipedia.org/wiki/Cliff%27s_delta
    # O(n*m) if naive; use faster using ranks
    x = np.asarray(x)
    y = np.asarray(y)
    nx = len(x)
    ny = len(y)
    # Use broadcasting if manageable
    # If too big, use rank-based approximation
    if nx*ny <= 5_000_000:
        # direct
        gt = (x[:,None] > y[None,:]).sum()
        lt = (x[:,None] < y[None,:]).sum()
        return (gt - lt) / (nx*ny)
    # rank-based approach
    # compute U statistic via ranks
    combined = np.concatenate([x,y])
    ranks = stats.rankdata(combined)
    rx = ranks[:nx].sum()
    # U for x
    U = rx - nx*(nx+1)/2
    # delta = (2U)/(nx*ny) -1
    return (2*U)/(nx*ny) - 1

cliff = cliffs_delta(g1, g0)

# simple linear regression controlling for page_id and num_words (and device) using statsmodels
import statsmodels.formula.api as smf

# Use log_speed to reduce skew
dys['reader_view'] = dys['reader_view'].astype(int)

# Build model
model = smf.ols('log_speed ~ reader_view + num_words + C(page_id) + C(device) + age + correct_rate', data=dys).fit()

print('n_total', n_total)
print('n_dys', n_dys)
print('min_speed', min_speed)
print('counts', counts.to_dict())
print('desc')
print(desc)
print('log_desc')
print(log_desc)
print('mw', mw)
print('tt', tt)
print('cliff', cliff)
print('model_reader_view_coef', model.params.get('reader_view'))
print('model_reader_view_p', model.pvalues.get('reader_view'))
print('model_reader_view_ci', model.conf_int().loc['reader_view'].tolist())
print('model_r2', model.rsquared)

