import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats


df = pd.read_csv('teachingratings.csv')

# basic cleaning
print('rows', len(df))
print('missing per column')
print(df.isna().sum())

# correlation
pearson_r, pearson_p = stats.pearsonr(df['beauty'], df['eval'])
spearman_r, spearman_p = stats.spearmanr(df['beauty'], df['eval'])
print('pearson', pearson_r, pearson_p)
print('spearman', spearman_r, spearman_p)

# OLS with controls
formula = (
    'eval ~ beauty + age + students + allstudents + '
    'C(gender) + C(minority) + C(native) + C(tenure) + '
    'C(division) + C(credits)'
)

model = smf.ols(formula, data=df).fit(cov_type='HC3')
print(model.summary())

coef = model.params['beauty']
se = model.bse['beauty']
pval = model.pvalues['beauty']
ci = model.conf_int().loc['beauty'].tolist()

# standardized effect (beauty SD) -> eval units
beauty_sd = df['beauty'].std()
eval_sd = df['eval'].std()
std_effect = coef * beauty_sd / eval_sd

print('beauty_coef', coef)
print('beauty_se', se)
print('beauty_p', pval)
print('beauty_ci', ci)
print('beauty_sd', beauty_sd)
print('eval_sd', eval_sd)
print('std_effect', std_effect)

# Simple regression only beauty
simple = smf.ols('eval ~ beauty', data=df).fit(cov_type='HC3')
print(simple.summary())
print('simple_beauty_coef', simple.params['beauty'])
print('simple_beauty_p', simple.pvalues['beauty'])

# partial correlation via residuals
# residualize eval and beauty on controls, then correlate residuals
controls = ['age', 'students', 'allstudents', 'gender', 'minority', 'native', 'tenure', 'division', 'credits']

# residualize eval
formula_eval = (
    'eval ~ age + students + allstudents + '
    'C(gender) + C(minority) + C(native) + C(tenure) + '
    'C(division) + C(credits)'
)
formula_beauty = (
    'beauty ~ age + students + allstudents + '
    'C(gender) + C(minority) + C(native) + C(tenure) + '
    'C(division) + C(credits)'
)
resid_eval = smf.ols(formula_eval, data=df).fit().resid
resid_beauty = smf.ols(formula_beauty, data=df).fit().resid
partial_r, partial_p = stats.pearsonr(resid_beauty, resid_eval)
print('partial_r', partial_r, partial_p)

# show counts for categories
for col in ['gender','minority','native','tenure','division','credits']:
    print(col)
    print(df[col].value_counts())
