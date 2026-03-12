import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# Load data

df = pd.read_csv('mortgage.csv')

# Focus on relevant columns for mortgage approvals
cols = [
    'female', 'deny', 'black', 'housing_expense_ratio', 'self_employed',
    'married', 'mortgage_credit', 'consumer_credit', 'bad_history',
    'PI_ratio', 'loan_to_value', 'denied_PMI'
]

sub = df[cols].copy()

# Drop rows with missing female or deny (deny has no missing in summary)
sub = sub.dropna(subset=['female', 'deny'])

# Ensure binary coding is numeric 0/1
sub['female'] = sub['female'].astype(int)
sub['deny'] = sub['deny'].astype(int)

# Unadjusted denial rate difference and chi-square test
ct = pd.crosstab(sub['female'], sub['deny'])
# rows: female=0,1; columns: deny=0,1

# Compute denial rates
male_denial = ct.loc[0, 1] / ct.loc[0].sum()
female_denial = ct.loc[1, 1] / ct.loc[1].sum()

# Chi-square test of independence
chi2, p_chi2, dof, expected = stats.chi2_contingency(ct)

# Two-proportion z-test (female vs male)
count = np.array([ct.loc[1, 1], ct.loc[0, 1]])
obs = np.array([ct.loc[1].sum(), ct.loc[0].sum()])
prop = count / obs
p_pool = count.sum() / obs.sum()
se = np.sqrt(p_pool * (1 - p_pool) * (1/obs[0] + 1/obs[1]))
z = (prop[0] - prop[1]) / se
p_z = 2 * (1 - stats.norm.cdf(abs(z)))

# Adjusted logistic regression
reg_data = sub.dropna()
X = reg_data[[
    'female', 'black', 'housing_expense_ratio', 'self_employed', 'married',
    'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio',
    'loan_to_value', 'denied_PMI'
]]
X = sm.add_constant(X)

y = reg_data['deny']

logit_model = sm.Logit(y, X)
result = logit_model.fit(disp=0)

# Extract female coefficient
female_coef = result.params['female']
female_p = result.pvalues['female']

# Odds ratio and 95% CI
or_female = np.exp(female_coef)
conf = result.conf_int().loc['female']
ci_low, ci_high = np.exp(conf[0]), np.exp(conf[1])

# Save results for later use
results = {
    'n_total': int(len(sub)),
    'n_male': int(ct.loc[0].sum()),
    'n_female': int(ct.loc[1].sum()),
    'male_denial_rate': float(male_denial),
    'female_denial_rate': float(female_denial),
    'denial_rate_diff_female_minus_male': float(female_denial - male_denial),
    'chi2_p': float(p_chi2),
    'z_p': float(p_z),
    'logit_female_coef': float(female_coef),
    'logit_female_or': float(or_female),
    'logit_female_or_ci_low': float(ci_low),
    'logit_female_or_ci_high': float(ci_high),
    'logit_female_p': float(female_p),
    'logit_n': int(len(reg_data))
}

with open('gender_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
