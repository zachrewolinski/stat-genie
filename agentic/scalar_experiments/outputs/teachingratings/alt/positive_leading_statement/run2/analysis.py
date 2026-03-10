import json
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

# Load data
path = 'teachingratings.csv'
df = pd.read_csv(path)

# Basic stats
n = len(df)

# Correlation
corr = df[['beauty','eval']].corr().iloc[0,1]

# Simple regression
model_simple = smf.ols('eval ~ beauty', data=df).fit(cov_type='HC3')

# Multiple regression with controls
# Use categorical factors
formula = (
    'eval ~ beauty + age + students + allstudents '
    '+ C(gender) + C(minority) + C(credits) + C(division) + C(native) + C(tenure)'
)
model_full = smf.ols(formula, data=df).fit(cov_type='HC3')

# Effect sizes
beta_simple = model_simple.params['beauty']
p_simple = model_simple.pvalues['beauty']

beta_full = model_full.params['beauty']
p_full = model_full.pvalues['beauty']

# Standardized effect (beauty -> eval) using z-scores
# Compute standardized beta via regression on z-scored vars
zdf = df.copy()
zdf['beauty_z'] = (zdf['beauty'] - zdf['beauty'].mean())/zdf['beauty'].std(ddof=0)
zdf['eval_z'] = (zdf['eval'] - zdf['eval'].mean())/zdf['eval'].std(ddof=0)
model_z = smf.ols('eval_z ~ beauty_z', data=zdf).fit(cov_type='HC3')
std_beta = model_z.params['beauty_z']

results = {
    'n': n,
    'corr': corr,
    'simple_beta': beta_simple,
    'simple_p': p_simple,
    'full_beta': beta_full,
    'full_p': p_full,
    'std_beta': std_beta,
    'simple_r2': model_simple.rsquared,
    'full_r2': model_full.rsquared,
}

print(json.dumps(results, indent=2))
