import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data
path = "mortgage.csv"
df = pd.read_csv(path)

# Basic cleaning: ensure numeric
# Use accept as outcome (1 accepted, 0 denied)

# Summary counts
n_total = len(df)

# Acceptance rates by gender
rate_by_gender = df.groupby('female')['accept'].mean()
count_by_gender = df['female'].value_counts().sort_index()

# Proportions test (chi-square)
contingency = pd.crosstab(df['female'], df['accept'])
chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency)

# Unadjusted logistic regression
model_unadj = smf.logit('accept ~ female', data=df).fit(disp=False)

# Adjusted logistic regression with key covariates
covariates = [
    'female', 'black', 'housing_expense_ratio', 'self_employed', 'married',
    'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio',
    'loan_to_value', 'denied_PMI'
]
formula = 'accept ~ ' + ' + '.join(covariates)
model_adj = smf.logit(formula, data=df).fit(disp=False)

# Extract female effect
coef_unadj = model_unadj.params['female']
p_unadj = model_unadj.pvalues['female']
or_unadj = float(np.exp(coef_unadj))

coef_adj = model_adj.params['female']
p_adj = model_adj.pvalues['female']
or_adj = float(np.exp(coef_adj))

# 95% CI for odds ratio
ci_unadj = model_unadj.conf_int().loc['female']
ci_adj = model_adj.conf_int().loc['female']
or_ci_unadj = np.exp(ci_unadj)
or_ci_adj = np.exp(ci_adj)

results = {
    'n_total': n_total,
    'rate_by_gender': {str(k): float(v) for k, v in rate_by_gender.items()},
    'count_by_gender': {str(k): int(v) for k, v in count_by_gender.items()},
    'chi2': float(chi2),
    'p_chi2': float(p_chi2),
    'unadj': {
        'coef': float(coef_unadj),
        'p': float(p_unadj),
        'odds_ratio': or_unadj,
        'or_ci_low': float(or_ci_unadj[0]),
        'or_ci_high': float(or_ci_unadj[1]),
    },
    'adj': {
        'coef': float(coef_adj),
        'p': float(p_adj),
        'odds_ratio': or_adj,
        'or_ci_low': float(or_ci_adj[0]),
        'or_ci_high': float(or_ci_adj[1]),
    },
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
