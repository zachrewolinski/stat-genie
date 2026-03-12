import math
import pandas as pd
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('teachingratings.csv')

# Ensure categorical variables
cat_cols = ['minority', 'gender', 'credits', 'division', 'native', 'tenure']
for c in cat_cols:
    if c in _df.columns:
        _df[c] = _df[c].astype('category')

# Simple correlation
corr = _df['beauty'].corr(_df['eval'])

# Simple OLS
model_simple = smf.ols('eval ~ beauty', data=_df).fit(cov_type='HC3')

# OLS with controls
_df['log_students'] = _df['students'].clip(lower=1).apply(math.log)
_df['log_allstudents'] = _df['allstudents'].clip(lower=1).apply(math.log)

model_controls = smf.ols(
    'eval ~ beauty + age + gender + minority + native + tenure + division + credits + log_students + log_allstudents',
    data=_df
).fit(cov_type='HC3')

# Professor fixed effects (within-professor)
_df['prof'] = _df['prof'].astype('category')
model_prof_fe = smf.ols('eval ~ beauty + C(prof)', data=_df).fit(cov_type='HC3')

# Output key stats
print('N:', len(_df))
print('Correlation beauty-eval:', corr)

print('\nSimple OLS:')
print(model_simple.summary().tables[1])

print('\nControls OLS:')
print(model_controls.summary().tables[1])

# FE model: just coefficient line to avoid robust F-test issues in summary
coef = model_prof_fe.params.get('beauty')
se = model_prof_fe.bse.get('beauty')
pval = model_prof_fe.pvalues.get('beauty')
print('\nProf FE OLS (beauty coefficient only):')
print(f'beauty coef prof_fe: {coef:.4f} (SE {se:.4f}) p={pval:.4g}')

# Extract coefficients for beauty in each model
for name, model in [('simple', model_simple), ('controls', model_controls), ('prof_fe', model_prof_fe)]:
    coef = model.params.get('beauty')
    se = model.bse.get('beauty')
    pval = model.pvalues.get('beauty')
    print(f'beauty coef {name}: {coef:.4f} (SE {se:.4f}) p={pval:.4g}')
