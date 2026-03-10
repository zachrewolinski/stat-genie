import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('teachingratings.csv')

# Core variables for research question
cols = [
    'beauty', 'eval', 'age', 'gender', 'minority', 'native',
    'tenure', 'division', 'credits', 'students', 'allstudents'
]

# Ensure columns exist
missing = [c for c in cols if c not in _df.columns]
if missing:
    raise SystemExit(f"Missing columns: {missing}")

# Drop rows with missing data in used columns
_df = _df[cols].dropna().copy()

# Standardize types for categorical columns
cat_cols = ['gender', 'minority', 'native', 'tenure', 'division', 'credits']
for c in cat_cols:
    _df[c] = _df[c].astype('category')

# Basic correlation
corr = _df['beauty'].corr(_df['eval'])

# Simple OLS: eval ~ beauty
m1 = smf.ols('eval ~ beauty', data=_df).fit()

# Multivariate OLS with controls
formula = 'eval ~ beauty + age + students + allstudents + C(gender) + C(minority) + C(native) + C(tenure) + C(division) + C(credits)'
m2 = smf.ols(formula, data=_df).fit()

# Extract key results
result = {
    'n': int(m2.nobs),
    'corr_beauty_eval': corr,
    'm1_beauty_coef': m1.params['beauty'],
    'm1_beauty_p': m1.pvalues['beauty'],
    'm1_r2': m1.rsquared,
    'm2_beauty_coef': m2.params['beauty'],
    'm2_beauty_p': m2.pvalues['beauty'],
    'm2_r2': m2.rsquared,
    'm2_beauty_ci_low': m2.conf_int().loc['beauty', 0],
    'm2_beauty_ci_high': m2.conf_int().loc['beauty', 1],
}

print(result)
