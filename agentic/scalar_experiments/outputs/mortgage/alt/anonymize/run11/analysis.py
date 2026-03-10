import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest

# Load data

df = pd.read_csv('mortgage.csv')

# Define columns
outcome = 'feature14'  # 1 if accepted
female = 'feature2'    # 1 if female

# Columns for adjusted model (exclude feature1 unique id, feature11 denial)
adj_cols = [
    female, 'feature3', 'feature4', 'feature5', 'feature6',
    'feature7', 'feature8', 'feature9', 'feature10', 'feature12', 'feature13'
]

# Drop missing
use_cols = [outcome] + adj_cols
work = df[use_cols].dropna().copy()

# Basic counts
n_total = len(work)

# Unadjusted approval rates by gender
rates = work.groupby(female)[outcome].mean()
counts = work.groupby(female)[outcome].agg(['sum', 'count'])

# Two-proportion z-test (acceptance rates)
# group 0 = male, group 1 = female
if list(counts.index) != [0, 1]:
    counts = counts.reindex([0, 1])
    rates = rates.reindex([0, 1])

successes = counts['sum'].values
nobs = counts['count'].values

zstat, pval_prop = proportions_ztest(successes, nobs)

# Adjusted logistic regression (GLM Binomial with robust SE)
X = work[adj_cols]
X = sm.add_constant(X, has_constant='add')

y = work[outcome]

glm_model = sm.GLM(y, X, family=sm.families.Binomial())
glm_result = glm_model.fit(cov_type='HC1')

coef = float(glm_result.params[female])
se = float(glm_result.bse[female])
pval = float(glm_result.pvalues[female])

odds_ratio = float(np.exp(coef))

# Average marginal effect (counterfactual prediction difference)
params = glm_result.params
X_female1 = X.copy()
X_female0 = X.copy()
X_female1[female] = 1
X_female0[female] = 0

lin1 = np.dot(X_female1, params)
lin0 = np.dot(X_female0, params)

p1 = 1 / (1 + np.exp(-lin1))
p0 = 1 / (1 + np.exp(-lin0))

ame = float(np.mean(p1 - p0))

output = {
    'n_total': int(n_total),
    'approval_rate_male': float(rates.loc[0]),
    'approval_rate_female': float(rates.loc[1]),
    'rate_diff_female_minus_male': float(rates.loc[1] - rates.loc[0]),
    'prop_test_z': float(zstat),
    'prop_test_p': float(pval_prop),
    'logit_coef_female_hc1': coef,
    'logit_se_female_hc1': se,
    'logit_p_female_hc1': pval,
    'odds_ratio_female': odds_ratio,
    'avg_marginal_effect_female': ame
}

print(json.dumps(output, indent=2))
