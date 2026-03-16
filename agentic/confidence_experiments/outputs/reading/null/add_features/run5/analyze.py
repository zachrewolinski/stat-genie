import pandas as pd
import numpy as np
from scipy import stats

# Load data
path = 'reading.csv'
df = pd.read_csv(path)

print('rows', len(df))
print('cols', df.columns.tolist())

# Determine dyslexia indicator
# use dyslexia_bin if exists, else dyslexia>0
if 'dyslexia_bin' in df.columns:
    dys_col = 'dyslexia_bin'
    dys = df[dys_col] == 1
elif 'dyslexia' in df.columns:
    dys_col = 'dyslexia'
    dys = df[dys_col] > 0
else:
    raise ValueError('No dyslexia column')

print('dys col', dys_col, 'count', dys.sum())

# ensure reader_view column
if 'reader_view' not in df.columns:
    raise ValueError('No reader_view column')

# speed column
speed_col = 'speed'
print('speed missing', df[speed_col].isna().sum())

# Subset dyslexia participants
sub = df[dys].copy()
print('sub rows', len(sub))

# Basic group stats
for rv in [0,1]:
    g = sub[sub['reader_view']==rv][speed_col].dropna()
    print('reader_view', rv, 'n', len(g), 'mean', g.mean(), 'median', g.median(), 'std', g.std())

# Welch t-test on raw speed
rv0 = sub[sub['reader_view']==0][speed_col].dropna()
rv1 = sub[sub['reader_view']==1][speed_col].dropna()

if len(rv0)>1 and len(rv1)>1:
    tstat, pval = stats.ttest_ind(rv1, rv0, equal_var=False)
    print('welch t raw', tstat, pval)

# log transform to handle skew (add small constant)
# use positive speed
sub_pos = sub[sub[speed_col]>0].copy()
sub_pos['log_speed'] = np.log(sub_pos[speed_col])
rv0_log = sub_pos[sub_pos['reader_view']==0]['log_speed']
rv1_log = sub_pos[sub_pos['reader_view']==1]['log_speed']

if len(rv0_log)>1 and len(rv1_log)>1:
    tstat_log, pval_log = stats.ttest_ind(rv1_log, rv0_log, equal_var=False)
    print('welch t log', tstat_log, pval_log)

# effect size (Cohen's d) for log speed
# compute d = (mean1-mean0)/pooled_sd (use pooled based on sample sizes)

n1, n0 = len(rv1_log), len(rv0_log)
mean1, mean0 = rv1_log.mean(), rv0_log.mean()
var1, var0 = rv1_log.var(ddof=1), rv0_log.var(ddof=1)
pooled_sd = np.sqrt(((n1-1)*var1 + (n0-1)*var0)/(n1+n0-2))
if pooled_sd>0:
    d = (mean1-mean0)/pooled_sd
else:
    d = np.nan
print('log speed mean1 mean0 diff', mean1, mean0, mean1-mean0, 'd', d)

# Non-parametric test
if len(rv0)>1 and len(rv1)>1:
    ustat, pval_u = stats.mannwhitneyu(rv1, rv0, alternative='two-sided')
    print('mannwhitney raw', ustat, pval_u)

# Regression controlling for page id and words etc? We'll do linear regression on log speed
import statsmodels.api as sm

# Build regression with controls: reader_view, page_id, num_words, Flesch_Kincaid, device, age, gender, education, language, english_native, retake_trial
# Only include columns that exist
controls = ['page_id', 'num_words', 'Flesch_Kincaid', 'device', 'age', 'gender', 'education', 'language', 'english_native', 'retake_trial']

cols = ['reader_view'] + [c for c in controls if c in sub_pos.columns]

# drop rows with missing values in these columns
reg_df = sub_pos[cols + ['log_speed']].dropna()
print('reg rows', len(reg_df), 'cols', cols)

# Create design matrix with categorical encoding
X = pd.get_dummies(reg_df[cols], drop_first=True)
X = sm.add_constant(X)
y = reg_df['log_speed']

model = sm.OLS(y, X).fit()
print(model.summary().tables[1])

# Extract reader_view coefficient
rv_coef = model.params.get('reader_view', np.nan)
rv_pval = model.pvalues.get('reader_view', np.nan)
print('reg reader_view coef', rv_coef, 'p', rv_pval)
