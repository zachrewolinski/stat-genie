import pandas as pd
import numpy as np
from scipy import stats

# Load data
path = 'reading.csv'
df = pd.read_csv(path)

# Define groups
speed = df['feature20']  # reading speed (WPM)
reader_view = df['feature3']  # 1 if reader view

dyslexia_binary = df['feature17']  # 1 if dyslexia

df = df.copy()
df['speed'] = speed

# Filter dyslexic individuals
subset = df[df['feature17'] == 1]

# Group by reader view
rv_on = subset[subset['feature3'] == 1]['speed']
rv_off = subset[subset['feature3'] == 0]['speed']

# Descriptive stats
stats_desc = {
    'n_total_dyslexia': subset.shape[0],
    'n_rv_on': rv_on.shape[0],
    'n_rv_off': rv_off.shape[0],
    'mean_rv_on': rv_on.mean(),
    'mean_rv_off': rv_off.mean(),
    'median_rv_on': rv_on.median(),
    'median_rv_off': rv_off.median(),
    'std_rv_on': rv_on.std(ddof=1),
    'std_rv_off': rv_off.std(ddof=1),
}
print('desc', stats_desc)

# Welch's t-test

t_stat, p_val = stats.ttest_ind(rv_on, rv_off, equal_var=False, nan_policy='omit')
print('welch t-test', t_stat, p_val)

# Mann-Whitney U test
u_stat, p_u = stats.mannwhitneyu(rv_on, rv_off, alternative='two-sided')
print('mannwhitney', u_stat, p_u)

# Cohen's d (Hedges g) for effect size

def cohens_d(x, y):
    x = x.dropna()
    y = y.dropna()
    nx, ny = len(x), len(y)
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    # pooled SD
    s = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    return (x.mean() - y.mean()) / s if s != 0 else np.nan


def hedges_g(x, y):
    x = x.dropna()
    y = y.dropna()
    nx, ny = len(x), len(y)
    d = cohens_d(x, y)
    # small sample correction
    correction = 1 - (3 / (4 * (nx + ny) - 9))
    return d * correction

print('cohens d', cohens_d(rv_on, rv_off))
print('hedges g', hedges_g(rv_on, rv_off))

# Also define dyslexia status with feature12>0 (includes severe)
subset2 = df[df['feature12'].fillna(0) > 0]
rv_on2 = subset2[subset2['feature3'] == 1]['speed']
rv_off2 = subset2[subset2['feature3'] == 0]['speed']
print('desc2', {
    'n_total_dyslexia12': subset2.shape[0],
    'n_rv_on': rv_on2.shape[0],
    'n_rv_off': rv_off2.shape[0],
    'mean_rv_on': rv_on2.mean(),
    'mean_rv_off': rv_off2.mean(),
    'median_rv_on': rv_on2.median(),
    'median_rv_off': rv_off2.median(),
})
print('welch t-test2', stats.ttest_ind(rv_on2, rv_off2, equal_var=False, nan_policy='omit'))
print('mannwhitney2', stats.mannwhitneyu(rv_on2, rv_off2, alternative='two-sided'))
print('hedges g2', hedges_g(rv_on2, rv_off2))

# Simple regression controlling for page words and language? Use OLS for subset
import statsmodels.api as sm

# build regression for dyslexia group
reg_df = subset[['speed','feature3','feature7','feature19']].dropna()
X = reg_df[['feature3','feature7','feature19']]
X = sm.add_constant(X)
model = sm.OLS(reg_df['speed'], X).fit()
print(model.summary().as_text())

