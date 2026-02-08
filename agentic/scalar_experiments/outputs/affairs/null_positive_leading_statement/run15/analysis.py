import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


df = pd.read_csv('affairs.csv')

# Basic group stats
summary = df.groupby('children')['affairs'].agg(['count','mean','median','std'])
summary_any = df.assign(any_affair=df['affairs']>0).groupby('children')['any_affair'].mean()

# OLS with controls (children yes vs no). Make children binary: yes=1

df['children_yes'] = (df['children'].str.lower()=='yes').astype(int)

# OLS
ols = smf.ols('affairs ~ children_yes + age + yearsmarried + C(gender) + religiousness + education + occupation + rating', data=df).fit(cov_type='HC3')

# Poisson regression for count with robust SE
poisson = smf.glm('affairs ~ children_yes + age + yearsmarried + C(gender) + religiousness + education + occupation + rating',
                  data=df, family=sm.families.Poisson()).fit(cov_type='HC3')

# Logistic for any affair
logit = smf.logit('any_affair ~ children_yes + age + yearsmarried + C(gender) + religiousness + education + occupation + rating',
                  data=df.assign(any_affair=(df['affairs']>0).astype(int))).fit(disp=False)

# Extract effects
ols_coef = ols.params['children_yes']
ols_p = ols.pvalues['children_yes']

poisson_coef = poisson.params['children_yes']
poisson_p = poisson.pvalues['children_yes']

logit_coef = logit.params['children_yes']
logit_p = logit.pvalues['children_yes']

# odds ratio
logit_or = float(np.exp(logit_coef))

# Prepare results
print('group_summary')
print(summary)
print('\nany_affair_rate')
print(summary_any)

print('\nOLS children_yes coef', ols_coef, 'p', ols_p)
print('Poisson children_yes coef', poisson_coef, 'p', poisson_p)
print('Logit children_yes coef', logit_coef, 'p', logit_p, 'OR', logit_or)

# Save key stats to json for later if needed
results = {
    'group_summary': summary.to_dict(),
    'any_affair_rate': summary_any.to_dict(),
    'ols_coef': float(ols_coef),
    'ols_p': float(ols_p),
    'poisson_coef': float(poisson_coef),
    'poisson_p': float(poisson_p),
    'logit_coef': float(logit_coef),
    'logit_p': float(logit_p),
    'logit_or': float(logit_or),
}

import json
with open('analysis_results.json','w') as f:
    json.dump(results,f,indent=2)
