import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm


df = pd.read_csv('reading.csv')

# Derived reading speed using reading time excluding scrolling (feature5)
# speed in words per minute
speed = df['feature7'] / (df['feature5'] / 60000.0)

df = df.assign(speed_wpm=speed)

# Dyslexic subset
sub = df[df['feature17'] == 1].copy()

rv = sub[sub['feature3'] == 1]['speed_wpm'].dropna()
no = sub[sub['feature3'] == 0]['speed_wpm'].dropna()

n_rv = len(rv)
n_no = len(no)
print('n_rv', n_rv, 'n_no', n_no)
print('mean_rv', rv.mean(), 'mean_no', no.mean())
print('median_rv', rv.median(), 'median_no', no.median())

# Welch t-test
res = stats.ttest_ind(rv, no, equal_var=False)
print('t', res.statistic, 'p', res.pvalue)

# effect size (Cohen d)

def cohens_d(x, y):
    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    s = np.sqrt(((nx-1)*vx + (ny-1)*vy) / (nx+ny-2))
    return (np.mean(x)-np.mean(y)) / s if s>0 else np.nan

print('cohen_d', cohens_d(rv, no))

# 95% CI for mean difference (Welch)
# use t distribution with Welch-Satterthwaite df
vx, vy = rv.var(ddof=1), no.var(ddof=1)
mean_diff = rv.mean() - no.mean()
se = np.sqrt(vx/n_rv + vy/n_no)

df_welch = (vx/n_rv + vy/n_no)**2 / ((vx**2)/((n_rv**2)*(n_rv-1)) + (vy**2)/((n_no**2)*(n_no-1)))

alpha = 0.05
t_crit = stats.t.ppf(1 - alpha/2, df_welch)
ci_low = mean_diff - t_crit*se
ci_high = mean_diff + t_crit*se
print('mean_diff', mean_diff, '95% CI', (ci_low, ci_high), 'df', df_welch)

# Robust checks: log transform and Mann-Whitney
rv_log = np.log(rv)
no_log = np.log(no)
res_log = stats.ttest_ind(rv_log, no_log, equal_var=False)
print('log t', res_log.statistic, 'p', res_log.pvalue)

u, p_u = stats.mannwhitneyu(rv, no, alternative='two-sided')
print('MW p', p_u)

# Regression controlling for covariates
sub = sub.dropna(subset=['speed_wpm','feature3','feature7','feature19','feature16','feature11','feature15'])
sub['log_speed'] = np.log(sub['speed_wpm'])
X = pd.get_dummies(sub[['feature3','feature7','feature19','feature16','feature11','feature15']], drop_first=True)
X = sm.add_constant(X)
model = sm.OLS(sub['log_speed'], X).fit()
coef = model.params.get('feature3')
pval = model.pvalues.get('feature3')
print('reg coef', coef, 'p', pval)
