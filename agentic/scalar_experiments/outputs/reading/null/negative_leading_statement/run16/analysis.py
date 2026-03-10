import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf


df = pd.read_csv('reading.csv')

# Define dyslexia subset
if 'dyslexia_bin' in df.columns:
    dys = df[df['dyslexia_bin'] == 1].copy()
else:
    dys = df[df['dyslexia'] > 0].copy()

# Basic counts
counts = dys['reader_view'].value_counts(dropna=False).sort_index()

# Descriptive stats
summary = dys.groupby('reader_view')['speed'].agg(['count','mean','median','std'])

# Welch t-test
rv1 = dys[dys['reader_view'] == 1]['speed'].dropna()
rv0 = dys[dys['reader_view'] == 0]['speed'].dropna()

t_res = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')

# Mann-Whitney (nonparam)
try:
    mw_res = stats.mannwhitneyu(rv1, rv0, alternative='two-sided')
except ValueError:
    mw_res = None

# Effect size (Cohen's d, Hedges g)

def cohen_d(x, y):
    x = x.dropna()
    y = y.dropna()
    nx = len(x); ny = len(y)
    if nx < 2 or ny < 2:
        return np.nan
    sx = x.std(ddof=1)
    sy = y.std(ddof=1)
    s = np.sqrt(((nx-1)*sx**2 + (ny-1)*sy**2) / (nx+ny-2))
    return (x.mean()-y.mean())/s if s>0 else np.nan


def hedges_g(x, y):
    d = cohen_d(x, y)
    nx = len(x.dropna()); ny = len(y.dropna())
    if np.isnan(d) or nx+ny<=2:
        return np.nan
    # small sample correction
    j = 1 - (3/(4*(nx+ny)-9))
    return d*j


d = cohen_d(rv1, rv0)

g = hedges_g(rv1, rv0)

# Regression with controls and clustered SEs by uuid
# Use log speed to reduce skew
# Remove nonpositive speeds (if any)
reg_df = dys.copy()
reg_df = reg_df[reg_df['speed'] > 0].copy()
reg_df['log_speed'] = np.log(reg_df['speed'])

# Categorical controls
# Use page_id, num_words, device, age, gender, education, language, english_native
# Keep rows with non-missing reader_view and log_speed
needed_cols = ['reader_view','log_speed','page_id','num_words','device','age','gender','education','language','english_native','uuid']
reg_df = reg_df.dropna(subset=needed_cols)

formula = 'log_speed ~ reader_view + C(page_id) + num_words + C(device) + age + C(gender) + C(education) + C(language) + C(english_native)'

model = smf.ols(formula, data=reg_df).fit(cov_type='cluster', cov_kwds={'groups': reg_df['uuid']})

# Report
print('Dyslexia subset size:', len(dys))
print('Reader_view counts:\n', counts)
print('\nSpeed summary by reader_view (dyslexia):\n', summary)
print('\nWelch t-test:', t_res)
print('Mann-Whitney:', mw_res)
print('Cohen d:', d)
print('Hedges g:', g)
print('\nRegression (log_speed) coef for reader_view:')
print(model.params.get('reader_view'))
print('p-value:', model.pvalues.get('reader_view'))
print('95% CI:', model.conf_int().loc['reader_view'].tolist())
