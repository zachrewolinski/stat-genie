import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('reading.csv')

# Define dyslexia subset (any dyslexia)
dys = _df[(
    (_df['dyslexia_bin'] == 1) | (_df['dyslexia'].fillna(0) > 0)
)].copy()

# Basic counts
print('Total rows:', len(_df))
print('Dyslexia rows:', len(dys))
print('Reader view counts (dyslexia):')
print(dys['reader_view'].value_counts(dropna=False))

# Drop missing speed
speed = dys['speed']
print('Missing speed:', speed.isna().sum())

# Ensure reader_view is binary
print('Reader view unique:', dys['reader_view'].dropna().unique())

# Group stats
for rv in [0,1]:
    grp = dys.loc[dys['reader_view'] == rv, 'speed']
    print(f"reader_view={rv} n={grp.notna().sum()} mean={grp.mean():.4f} median={grp.median():.4f} std={grp.std():.4f}")

# Welch t-test on raw speed
rv0 = dys.loc[dys['reader_view'] == 0, 'speed'].dropna()
rv1 = dys.loc[dys['reader_view'] == 1, 'speed'].dropna()

tstat, pval = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')
print('Welch t-test raw speed: t=', tstat, 'p=', pval)

# Mann-Whitney U test (nonparametric)
try:
    ustat, upval = stats.mannwhitneyu(rv1, rv0, alternative='two-sided')
    print('Mann-Whitney U: U=', ustat, 'p=', upval)
except Exception as e:
    print('Mann-Whitney failed:', e)

# Log-transform speed (positive)
# Avoid zero or negative
log_speed = np.log(dys['speed'])

rv0_log = log_speed[dys['reader_view']==0].dropna()
rv1_log = log_speed[dys['reader_view']==1].dropna()

tstat_log, pval_log = stats.ttest_ind(rv1_log, rv0_log, equal_var=False, nan_policy='omit')
print('Welch t-test log speed: t=', tstat_log, 'p=', pval_log)

# Effect size (Cohen d) on raw and log

def cohens_d(x, y):
    x = x.dropna()
    y = y.dropna()
    nx = len(x)
    ny = len(y)
    if nx < 2 or ny < 2:
        return np.nan
    vx = x.var(ddof=1)
    vy = y.var(ddof=1)
    pooled = ((nx-1)*vx + (ny-1)*vy) / (nx + ny - 2)
    return (x.mean() - y.mean()) / np.sqrt(pooled)

print('Cohen d raw (rv1 - rv0):', cohens_d(rv1, rv0))
print('Cohen d log (rv1 - rv0):', cohens_d(rv1_log, rv0_log))

# Regression with controls
# Build formula; use C() for categoricals
formula = (
    "np.log(speed) ~ reader_view + num_words + C(page_id) + C(device) + age + "
    "C(gender) + C(education) + C(language) + C(english_native) + "
    "retake_trial + correct_rate + Flesch_Kincaid + img_width"
)

# Remove rows with missing needed columns
cols_needed = ['speed','reader_view','num_words','page_id','device','age','gender','education','language','english_native',
               'retake_trial','correct_rate','Flesch_Kincaid','img_width']
reg_df = dys[cols_needed].dropna().copy()

print('Regression rows:', len(reg_df))

if len(reg_df) > 50:
    model = smf.ols(formula=formula, data=reg_df).fit(cov_type='HC3')
    print(model.summary().tables[1])
    coef = model.params.get('reader_view', np.nan)
    p = model.pvalues.get('reader_view', np.nan)
    print('reader_view coef (log speed):', coef, 'p=', p)
else:
    print('Not enough rows for regression')

# Simple regression without many controls
model2 = smf.ols('np.log(speed) ~ reader_view', data=dys.dropna(subset=['speed','reader_view'])).fit(cov_type='HC3')
print(model2.summary().tables[1])
print('Simple log-speed coef:', model2.params.get('reader_view', np.nan), 'p=', model2.pvalues.get('reader_view', np.nan))
