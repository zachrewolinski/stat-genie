import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest

# Load data
path = 'mortgage.csv'
df = pd.read_csv(path)

print('rows', len(df))
print('columns', df.columns.tolist())

# Basic check of accept/deny
print('accept unique', df['accept'].unique())
print('deny unique', df['deny'].unique())
print('accept mean', df['accept'].mean())
print('deny mean', df['deny'].mean())
print('accept+deny unique', (df['accept'] + df['deny']).unique()[:5])

# missing values
print('missing counts (top 10)')
print(df.isna().sum().sort_values(ascending=False).head(10))

# Basic approval rates by gender
rate_by_gender = df.groupby('female')['accept'].mean()
count_by_gender = df.groupby('female')['accept'].count()
print('rate_by_gender', rate_by_gender.to_dict())
print('count_by_gender', count_by_gender.to_dict())

# Two-proportion z-test (accept rates)
counts = df.groupby('female')['accept'].sum()
ns = df.groupby('female')['accept'].count()
stat, pval = proportions_ztest(count=counts.values, nobs=ns.values, alternative='two-sided')
print('two-proportion z-test stat', stat, 'p', pval)

# Logistic regression: accept ~ female (bivariate)
X = sm.add_constant(df[['female']])
model = sm.Logit(df['accept'], X, missing='drop')
res = model.fit(disp=False)
print('logit bivariate', res.params, res.pvalues)

# Multivariate logistic regression with key mortgage-related covariates
covariates = ['female','black','housing_expense_ratio','self_employed','married','mortgage_credit',
              'consumer_credit','bad_history','PI_ratio','loan_to_value','denied_PMI']
# Keep only columns that exist
covariates = [c for c in covariates if c in df.columns]

# Drop rows with missing in covariates or outcome
df_model = df[covariates + ['accept']].dropna()
X2 = sm.add_constant(df_model[covariates])
model2 = sm.Logit(df_model['accept'], X2)
res2 = model2.fit(disp=False)
print('multivariate n', len(df_model))
print('logit multivariate params')
print(res2.params)
print('pvalues')
print(res2.pvalues)

# Odds ratio and CI for female
female_coef = res2.params['female']
se = res2.bse['female']
odds_ratio = np.exp(female_coef)
ci_low = np.exp(female_coef - 1.96*se)
ci_high = np.exp(female_coef + 1.96*se)
print('female OR', odds_ratio, 'CI', (ci_low, ci_high))

# Also compute marginal effect of female from multivariate model
margeff = res2.get_margeff(at='overall', method='dydx')
print('marginal effects')
print(margeff.summary())

# Save key outputs to a JSON for later
import json
output = {
    'rate_by_gender': rate_by_gender.to_dict(),
    'count_by_gender': count_by_gender.to_dict(),
    'ztest_stat': float(stat),
    'ztest_pvalue': float(pval),
    'bivar_coef': res.params['female'],
    'bivar_pvalue': res.pvalues['female'],
    'multi_coef': float(female_coef),
    'multi_pvalue': float(res2.pvalues['female']),
    'multi_or': float(odds_ratio),
    'multi_or_ci': [float(ci_low), float(ci_high)],
    'multi_margeff': float(margeff.margeff[covariates.index('female')]) if 'female' in covariates else None,
    'multi_margeff_se': float(margeff.margeff_se[covariates.index('female')]) if 'female' in covariates else None,
}
with open('analysis_results.json','w') as f:
    json.dump(output,f,indent=2)
print('saved analysis_results.json')
