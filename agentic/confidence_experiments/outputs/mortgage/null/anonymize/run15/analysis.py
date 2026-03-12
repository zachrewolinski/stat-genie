import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest


df = pd.read_csv('mortgage.csv')

# Define variables
female = df['feature2']
approved = df['feature14']  # 1 accepted, 0 denied

# Check complementarity with feature11 (denied)
comp_check = (df['feature11'] + df['feature14']).describe()

# Approval rates by gender
summary = df.groupby('feature2')['feature14'].agg(['mean', 'count'])
summary['approved'] = (summary['mean'] * summary['count']).round(0)

# Two-proportion z-test for approval rates
count = summary['approved'].values
nobs = summary['count'].values
zstat, pval = proportions_ztest(count, nobs)

# Unadjusted logistic regression
X_unadj = sm.add_constant(female)
model_unadj = sm.Logit(approved, X_unadj, missing='drop')
res_unadj = model_unadj.fit(disp=0)

# Adjusted logistic regression with all covariates
covariates = [
    'feature2',  # female
    'feature3',  # Black
    'feature4',  # housing expense ratio
    'feature5',  # self-employed
    'feature6',  # married
    'feature7',  # mortgage credit score
    'feature8',  # consumer credit score
    'feature9',  # bad credit history
    'feature10', # debt-to-income
    'feature12', # loan-to-value
    'feature13', # PMI denial
    'feature1'   # loan amount (unknown but numeric)
]

X_adj = sm.add_constant(df[covariates])
model_adj = sm.Logit(approved, X_adj, missing='drop')
res_adj = model_adj.fit(disp=0)

# Odds ratios for female
or_unadj = np.exp(res_unadj.params['feature2'])
or_adj = np.exp(res_adj.params['feature2'])

output = {
    'comp_check': comp_check.to_dict(),
    'summary': summary.to_dict(),
    'ztest': {'zstat': float(zstat), 'pval': float(pval)},
    'unadj': {
        'coef': float(res_unadj.params['feature2']),
        'pval': float(res_unadj.pvalues['feature2']),
        'or': float(or_unadj),
        'ci': [float(x) for x in np.exp(res_unadj.conf_int().loc['feature2'].values)],
    },
    'adj': {
        'coef': float(res_adj.params['feature2']),
        'pval': float(res_adj.pvalues['feature2']),
        'or': float(or_adj),
        'ci': [float(x) for x in np.exp(res_adj.conf_int().loc['feature2'].values)],
    },
    'model_adj_nobs': int(res_adj.nobs),
    'model_unadj_nobs': int(res_unadj.nobs),
}

print(output)
