import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

# Load data
df = pd.read_csv('teachingratings.csv')

# Basic cleaning: ensure categories
cat_cols = ['minority','gender','credits','division','native','tenure']
for c in cat_cols:
    df[c] = df[c].astype('category')

# Simple correlation
corr = df['beauty'].corr(df['eval'])

# OLS regression with controls
formula = 'eval ~ beauty + age + C(gender) + C(minority) + C(credits) + C(division) + C(native) + C(tenure) + students + allstudents'
model = smf.ols(formula, data=df).fit(cov_type='HC3')

# Extract beauty coefficient, p-value
coef = model.params['beauty']
pval = model.pvalues['beauty']

# effect of 1 SD beauty on eval
sd_beauty = df['beauty'].std()
std_effect = coef * sd_beauty

# standardized coefficient (beta) using standard deviations
sd_eval = df['eval'].std()
std_beta = (coef * sd_beauty) / sd_eval

# Simple bivariate regression to see raw effect
biv = smf.ols('eval ~ beauty', data=df).fit(cov_type='HC3')

out = {
    'n': len(df),
    'corr': corr,
    'coef': coef,
    'pval': pval,
    'std_effect': std_effect,
    'std_beta': std_beta,
    'biv_coef': biv.params['beauty'],
    'biv_pval': biv.pvalues['beauty'],
}

print(out)
print(model.summary())
