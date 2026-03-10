import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'reading.csv'
df = pd.read_csv(path)

# Identify dyslexia indicator
if 'dyslexia_bin' in df.columns:
    dyslexia_mask = df['dyslexia_bin'] == 1
else:
    dyslexia_mask = df['dyslexia'].isin([1,2])

# Subset to dyslexic participants
sub = df[dyslexia_mask].copy()

# Basic counts
n_total = len(sub)

# Reader view groups
rv0 = sub[sub['reader_view'] == 0]
rv1 = sub[sub['reader_view'] == 1]

# Function to compute summary

def summary_stats(series):
    return {
        'n': int(series.notna().sum()),
        'mean': float(series.mean()),
        'median': float(series.median()),
        'std': float(series.std(ddof=1)),
    }

speed_col = 'speed'

summaries = {
    'rv0': summary_stats(rv0[speed_col]),
    'rv1': summary_stats(rv1[speed_col])
}

# Welch t-test
welch = stats.ttest_ind(rv1[speed_col], rv0[speed_col], equal_var=False, nan_policy='omit')

# Mann-Whitney U
try:
    mw = stats.mannwhitneyu(rv1[speed_col], rv0[speed_col], alternative='two-sided')
except Exception as e:
    mw = None

# Effect size (Cohen's d) for independent groups
# Use pooled SD (Welch's d) with unequal n
n1 = summaries['rv1']['n']
n0 = summaries['rv0']['n']
mean1 = summaries['rv1']['mean']
mean0 = summaries['rv0']['mean']
std1 = summaries['rv1']['std']
std0 = summaries['rv0']['std']

# Compute pooled SD
pooled_sd = np.sqrt(((n1-1)*std1**2 + (n0-1)*std0**2) / (n1 + n0 - 2)) if (n1+n0-2)>0 else np.nan
cohen_d = (mean1 - mean0) / pooled_sd if pooled_sd > 0 else np.nan

# Check whether within-subject (paired) is possible by uuid
# Compute per-uuid means by reader_view, only for uuids with both conditions
if 'uuid' in sub.columns:
    per_uuid = sub.groupby(['uuid','reader_view'])[speed_col].mean().unstack('reader_view')
    both = per_uuid.dropna()
    paired_n = len(both)
    if paired_n > 1:
        paired_t = stats.ttest_rel(both[1], both[0])
        paired_diff_mean = float((both[1]-both[0]).mean())
    else:
        paired_t = None
        paired_diff_mean = np.nan
else:
    paired_n = 0
    paired_t = None
    paired_diff_mean = np.nan

# Log-transform speed to reduce skew and re-run Welch
sub['log_speed'] = np.log(sub[speed_col])
rv0_log = sub[sub['reader_view']==0]['log_speed']
rv1_log = sub[sub['reader_view']==1]['log_speed']
welch_log = stats.ttest_ind(rv1_log, rv0_log, equal_var=False, nan_policy='omit')

# Regression with controls: speed ~ reader_view + page_id + num_words + device + age + gender + education + english_native
# Use robust SE (HC3). Only if columns exist.
controls = ['page_id','num_words','device','age','gender','education','english_native']
available_controls = [c for c in controls if c in sub.columns]

formula = 'speed ~ reader_view'
if available_controls:
    formula += ' + ' + ' + '.join([f'C({c})' if sub[c].dtype == 'object' or str(sub[c].dtype)=='category' else c for c in available_controls])

# Fit OLS
try:
    model = smf.ols(formula, data=sub).fit(cov_type='HC3')
    rv_coef = model.params.get('reader_view', np.nan)
    rv_p = model.pvalues.get('reader_view', np.nan)
except Exception as e:
    model = None
    rv_coef = np.nan
    rv_p = np.nan

# Output results
print('n_total', n_total)
print('n_rv0', summaries['rv0']['n'], 'n_rv1', summaries['rv1']['n'])
print('summary_rv0', summaries['rv0'])
print('summary_rv1', summaries['rv1'])
print('welch', welch)
print('mannwhitney', mw)
print('cohen_d', cohen_d)
print('paired_n', paired_n, 'paired_t', paired_t, 'paired_diff_mean', paired_diff_mean)
print('welch_log', welch_log)
print('formula', formula)
print('rv_coef', rv_coef, 'rv_p', rv_p)
