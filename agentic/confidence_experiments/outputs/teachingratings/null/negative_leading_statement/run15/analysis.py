import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

path = 'teachingratings.csv'
df = pd.read_csv(path)

# Basic info
n = len(df)
missing = df.isna().sum()

# Pearson and Spearman correlations
pearson_r, pearson_p = stats.pearsonr(df['beauty'], df['eval'])
spearman_r, spearman_p = stats.spearmanr(df['beauty'], df['eval'])

# Simple OLS
m1 = smf.ols('eval ~ beauty', data=df).fit()

# Controls: categorical factors and numeric covariates
# Use log(allstudents) and response rate to avoid collinearity with students
# Add age as numeric, plus categorical factors.

df = df.assign(
    log_allstudents=np.log(df['allstudents']),
    response_rate=df['students'] / df['allstudents']
)

formula = (
    'eval ~ beauty + age + log_allstudents + response_rate '
    '+ C(gender) + C(minority) + C(credits) + C(division) + C(native) + C(tenure)'
)

m2 = smf.ols(formula, data=df).fit(cov_type='HC3')

# Clustered SE by professor (optional robustness)
try:
    m3 = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['prof']})
except Exception as e:
    m3 = None


out = {
    'n': n,
    'missing': missing.to_dict(),
    'pearson_r': pearson_r,
    'pearson_p': pearson_p,
    'spearman_r': spearman_r,
    'spearman_p': spearman_p,
    'm1_coef': m1.params['beauty'],
    'm1_p': m1.pvalues['beauty'],
    'm1_r2': m1.rsquared,
    'm2_coef': m2.params['beauty'],
    'm2_p': m2.pvalues['beauty'],
    'm2_r2': m2.rsquared,
}

if m3 is not None:
    out.update({
        'm3_coef': m3.params['beauty'],
        'm3_p': m3.pvalues['beauty'],
        'm3_r2': m3.rsquared,
    })

# Print key output
print(out)
