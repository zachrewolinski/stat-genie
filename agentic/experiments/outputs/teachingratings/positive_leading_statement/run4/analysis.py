import pandas as pd
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('teachingratings.csv')

# Ensure categorical types
cat_cols = ['minority', 'gender', 'credits', 'division', 'native', 'tenure']
for c in cat_cols:
    df[c] = df[c].astype('category')

# Base model: eval ~ beauty
m1 = smf.ols('eval ~ beauty', data=df).fit(cov_type='HC3')

# Full model with controls
m2 = smf.ols(
    'eval ~ beauty + age + C(gender) + C(minority) + C(credits) + C(division) + C(native) + C(tenure) + students + allstudents',
    data=df
).fit(cov_type='HC3')

print('Model 1: eval ~ beauty (HC3 robust SE)')
print(m1.summary())
print('\nModel 2: eval ~ beauty + controls (HC3 robust SE)')
print(m2.summary())

# Key results for quick reference
for label, model in [('m1', m1), ('m2', m2)]:
    coef = model.params['beauty']
    pval = model.pvalues['beauty']
    print(f"\n{label} beauty coef: {coef:.4f}, p-value: {pval:.4g}")
