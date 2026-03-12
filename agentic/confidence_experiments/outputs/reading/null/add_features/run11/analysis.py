import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Load data
path = 'reading.csv'
df = pd.read_csv(path)

# Basic cleaning
# Use dyslexia_bin to identify dyslexic participants (1 = dyslexia)
df = df.copy()

# Filter dyslexic participants
if 'dyslexia_bin' in df.columns:
    dys = df[df['dyslexia_bin'] == 1].copy()
else:
    # fallback: dyslexia > 0
    dys = df[df['dyslexia'] > 0].copy()

# Keep necessary columns and drop missing
cols_needed = ['uuid', 'reader_view', 'speed', 'num_words', 'page_id']
for c in cols_needed:
    if c not in dys.columns:
        raise ValueError(f"Missing column: {c}")

# Drop missing or non-positive speeds
before = len(dys)
dys = dys.dropna(subset=['reader_view', 'speed'])
# Some extreme values may exist; keep positive speeds for log transform
# If speed is zero or negative (unlikely), drop

dys = dys[dys['speed'] > 0]

# Summary stats
n_rows = len(dys)
n_subjects = dys['uuid'].nunique()
counts = dys['reader_view'].value_counts().sort_index()

# Group summary
summary = dys.groupby('reader_view')['speed'].agg(['count', 'mean', 'median', 'std'])

# Log speed for parametric tests
log_speed = np.log(dys['speed'])

# Welch t-test on log speed between reader_view groups
rv0 = dys.loc[dys['reader_view'] == 0, 'speed']
rv1 = dys.loc[dys['reader_view'] == 1, 'speed']
log_rv0 = np.log(rv0)
log_rv1 = np.log(rv1)

ttest = stats.ttest_ind(log_rv1, log_rv0, equal_var=False, nan_policy='omit')

# Mann-Whitney U on raw speed (nonparametric)
try:
    mwu = stats.mannwhitneyu(rv1, rv0, alternative='two-sided')
except ValueError:
    mwu = None

# Effect size: Cohen's d on log speed
mean1 = log_rv1.mean()
mean0 = log_rv0.mean()
var1 = log_rv1.var(ddof=1)
var0 = log_rv0.var(ddof=1)
# Pooled SD for Welch? Use average of variances weighted by n
n1 = log_rv1.shape[0]
n0 = log_rv0.shape[0]
pooled_sd = np.sqrt(((n1 - 1) * var1 + (n0 - 1) * var0) / (n1 + n0 - 2)) if (n1 + n0 - 2) > 0 else np.nan
cohen_d = (mean1 - mean0) / pooled_sd if pooled_sd and not np.isnan(pooled_sd) else np.nan

# Percent difference in median speed
median0 = rv0.median()
median1 = rv1.median()
median_pct = (median1 - median0) / median0 * 100 if median0 != 0 else np.nan

# Regression with cluster-robust SE by uuid, controlling for page_id and num_words
# Use log(speed) as outcome
# Some page_id may be missing; drop rows with missing page_id/num_words
reg_df = dys.dropna(subset=['page_id', 'num_words'])
reg_df = reg_df.copy()
reg_df['log_speed'] = np.log(reg_df['speed'])

model = smf.ols('log_speed ~ reader_view + num_words + C(page_id)', data=reg_df).fit(
    cov_type='cluster', cov_kwds={'groups': reg_df['uuid']}
)

# Paired within-subject analysis: average speed per subject per condition
# Keep only subjects with both reader_view conditions
subj_means = dys.groupby(['uuid', 'reader_view'])['speed'].mean().unstack()
paired = subj_means.dropna()
paired_n = paired.shape[0]
if paired_n > 1:
    paired_log = np.log(paired)
    paired_t = stats.ttest_rel(paired_log[1], paired_log[0], nan_policy='omit')
    paired_diff = (paired_log[1] - paired_log[0]).mean()
else:
    paired_t = None
    paired_diff = np.nan

# Print results
print('Dyslexic subset rows:', n_rows)
print('Dyslexic subset subjects:', n_subjects)
print('Reader_view counts:', counts.to_dict())
print('\nGroup summary (speed):')
print(summary)
print('\nWelch t-test on log(speed):', ttest)
print('Cohen d (log speed):', cohen_d)
print('Median percent difference (speed):', median_pct)
if mwu is not None:
    print('Mann-Whitney U (speed):', mwu)
print('\nCluster-robust OLS (log_speed ~ reader_view + num_words + C(page_id))')
print(model.summary().tables[1])
print('\nPaired within-subject (mean speed per condition):')
print('paired_n:', paired_n)
if paired_t is not None:
    print('paired t-test on log(speed):', paired_t)
    print('mean paired log diff (rv1 - rv0):', paired_diff)

