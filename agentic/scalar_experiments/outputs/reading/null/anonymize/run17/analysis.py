import pandas as pd
import numpy as np
from scipy import stats

# Load data
path = 'reading.csv'

df = pd.read_csv(path)

# Basic info
print('shape', df.shape)
print(df.head())

# compute candidate speeds
# feature5: reading time minus scrolling (ms)
# feature4: time on page (ms)
# feature7: num words

# Avoid divide by zero
for time_col in ['feature4','feature5']:
    # words per minute
    df[f'speed_{time_col}'] = df['feature7'] / (df[time_col] / 60000.0)

# Compare with feature20
for col in ['speed_feature4','speed_feature5']:
    corr = df[col].corr(df['feature20'])
    print('corr', col, corr)

print('feature20 stats', df['feature20'].describe())

# Inspect dyslexia status: feature17 is dyslexia yes/no
print('dyslexia counts', df['feature17'].value_counts(dropna=False))

# Reader view: feature3
print('reader view counts', df['feature3'].value_counts(dropna=False))

# Focus on dyslexic participants (feature17 == 1)
dys = df[df['feature17']==1]

# For dyslexic, compare reading speed between reader view vs not
# We'll use speed_feature5 (if correlates with feature20)
for speed_col in ['speed_feature4','speed_feature5','feature20']:
    rv = dys[dys['feature3']==1][speed_col].dropna()
    no = dys[dys['feature3']==0][speed_col].dropna()
    print('\n', speed_col, 'dys RV n', len(rv), 'No n', len(no))
    print('means', rv.mean(), no.mean())
    # Welch t-test
    tstat, pval = stats.ttest_ind(rv, no, equal_var=False, nan_policy='omit')
    print('t', tstat, 'p', pval)
    # effect size (Cohen d) with pooled SD (using Welch) -> Hedges g maybe
    def cohens_d(x, y):
        nx, ny = len(x), len(y)
        vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
        # pooled sd (unbiased)
        s = np.sqrt(((nx-1)*vx + (ny-1)*vy) / (nx+ny-2))
        return (np.mean(x)-np.mean(y)) / s if s>0 else np.nan
    d = cohens_d(rv, no)
    print('cohen_d', d)

# Maybe use log speed due to outliers
for speed_col in ['speed_feature4','speed_feature5','feature20']:
    rv = dys[dys['feature3']==1][speed_col].dropna()
    no = dys[dys['feature3']==0][speed_col].dropna()
    # log transform
    rv_log = np.log(rv)
    no_log = np.log(no)
    tstat, pval = stats.ttest_ind(rv_log, no_log, equal_var=False, nan_policy='omit')
    print('\nlog', speed_col, 't', tstat, 'p', pval, 'means', rv_log.mean(), no_log.mean())

# Nonparametric test
for speed_col in ['speed_feature4','speed_feature5','feature20']:
    rv = dys[dys['feature3']==1][speed_col].dropna()
    no = dys[dys['feature3']==0][speed_col].dropna()
    u, p = stats.mannwhitneyu(rv, no, alternative='two-sided')
    print('\nMW', speed_col, 'u', u, 'p', p)

# optional: control for text complexity etc using regression
# We'll run linear regression with log speed for dyslexic participants.
import statsmodels.api as sm

# choose speed_feature5 as dependent maybe
speed_col = 'speed_feature5'

# predictors: reader view feature3, words count, readability, language maybe, device, page id
# We'll use simple model with reader view + words + readability + retake + device (categorical) + language

# Build design matrix
sub = dys.copy()
sub = sub.dropna(subset=[speed_col,'feature3','feature7','feature19','feature16','feature11','feature15'])

# log speed to reduce skew
sub['log_speed'] = np.log(sub[speed_col])

X = pd.get_dummies(sub[['feature3','feature7','feature19','feature16','feature11','feature15']], drop_first=True)
X = sm.add_constant(X)

y = sub['log_speed']

model = sm.OLS(y, X).fit()
print('\nOLS summary for dyslexic log speed (speed_feature5):')
print(model.summary().tables[1])

# Extract coefficient for feature3
coef = model.params.get('feature3', np.nan)
se = model.bse.get('feature3', np.nan)
print('feature3 coef', coef, 'se', se)
