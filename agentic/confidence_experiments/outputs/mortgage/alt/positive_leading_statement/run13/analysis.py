import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data
file_path = 'mortgage.csv'
df = pd.read_csv(file_path)

# Clean infinities and missing values for variables of interest
df = df.replace([np.inf, -np.inf], np.nan)

# Ensure relevant columns
# gender: female (1 if female, 0 if male)
# outcome: accept (1 accepted, 0 denied)

# Basic counts (after cleaning for gender/accept)
df_basic = df[['female', 'accept']].dropna()
n_total = len(df_basic)

# Acceptance rates by gender
rates = df_basic.groupby('female')['accept'].agg(['mean','count','sum'])

# Two-proportion z-test
male = df_basic[df_basic['female'] == 0]
female = df_basic[df_basic['female'] == 1]

# Proportions
p1 = female['accept'].mean()
p0 = male['accept'].mean()

n1 = female.shape[0]
n0 = male.shape[0]

# Two-proportion z-test
p_pool = (female['accept'].sum() + male['accept'].sum()) / (n1 + n0)
se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n0))
if se > 0:
    z = (p1 - p0) / se
    p_val_z = 2 * (1 - stats.norm.cdf(abs(z)))
else:
    z = np.nan
    p_val_z = np.nan

# Chi-square test of independence
contingency = pd.crosstab(df_basic['female'], df_basic['accept'])
chi2, p_chi, dof, expected = stats.chi2_contingency(contingency)

# Logistic regression (baseline: accept ~ female)
X1 = sm.add_constant(df_basic['female'])
model1 = sm.Logit(df_basic['accept'], X1)
res1 = model1.fit(disp=False)

# Logistic regression with controls
controls = [
    'female','black','housing_expense_ratio','self_employed','married',
    'mortgage_credit','consumer_credit','bad_history','PI_ratio','loan_to_value'
]

# drop rows with missing in controls/outcome
df_controls = df[controls + ['accept']].dropna()
X2 = sm.add_constant(df_controls[controls])
model2 = sm.Logit(df_controls['accept'], X2)
res2 = model2.fit(disp=False)

# Extract female coefficient from models
coef1 = res1.params['female']
p1_lr = res1.pvalues['female']
coef2 = res2.params['female']
p2_lr = res2.pvalues['female']
"""
LogitResults does not expose residuals needed for standard sandwich estimators
in this statsmodels build, so we rely on conventional MLE standard errors.
"""
p2_lr_rob = None

# Compute average marginal effect for female in model2
# Use statsmodels marginal effects
margeff = res2.get_margeff(at='overall')
me_table = margeff.summary_frame()
me_female = me_table.loc['female'] if 'female' in me_table.index else None

# Output results
output = {
    'n_total': n_total,
    'n_controls': len(df_controls),
    'rates_by_female': rates.to_dict(),
    'p1_female_accept': p1,
    'p0_male_accept': p0,
    'diff': p1 - p0,
    'two_prop_z': z,
    'p_val_two_prop': p_val_z,
    'chi2': chi2,
    'p_val_chi2': p_chi,
    'logit_female_only_coef': coef1,
    'logit_female_only_p': p1_lr,
    'logit_controls_coef': coef2,
    'logit_controls_p': p2_lr,
    'logit_controls_p_robust': p2_lr_rob,
    'marginal_effect_female': None if me_female is None else me_female.to_dict(),
}

print(json.dumps(output, indent=2, default=str))
