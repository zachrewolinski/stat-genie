import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Load data
path = "mortgage.csv"
df = pd.read_csv(path)

# Basic sanity: ensure binary columns are numeric
# Compute acceptance rates by gender
rate_by_gender = df.groupby('female')['accept'].mean()
count_by_gender = df.groupby('female')['accept'].count()

# Contingency table for chi-square (accept vs female)
cont_table = pd.crosstab(df['female'], df['accept'])
chi2, p, dof, expected = stats.chi2_contingency(cont_table)

# Logistic regression: accept ~ female
model_simple = smf.logit('accept ~ female', data=df).fit(disp=0)

# Logistic regression with controls
controls = ['black','housing_expense_ratio','self_employed','married','mortgage_credit','consumer_credit','bad_history','PI_ratio','loan_to_value','denied_PMI']
formula = 'accept ~ female + ' + ' + '.join(controls)
model_ctrl = smf.logit(formula, data=df).fit(disp=0)

# Extract key stats
simple_coef = model_simple.params['female']
simple_p = model_simple.pvalues['female']
ctrl_coef = model_ctrl.params['female']
ctrl_p = model_ctrl.pvalues['female']

# Odds ratios
simple_or = float(np.exp(simple_coef))
ctrl_or = float(np.exp(ctrl_coef))

# Effect size: difference in acceptance rates
rate_diff = rate_by_gender.loc[1] - rate_by_gender.loc[0]

# Print results for inspection
print("Counts by gender:")
print(count_by_gender)
print("Acceptance rates by gender:")
print(rate_by_gender)
print(f"Rate difference (female - male): {rate_diff:.4f}")
print("Chi-square test:")
print(f"chi2={chi2:.4f}, p={p:.6g}, dof={dof}")
print("Simple logit accept ~ female:")
print(f"coef={simple_coef:.4f}, OR={simple_or:.4f}, p={simple_p:.6g}")
print("Controlled logit accept ~ female + covariates:")
print(f"coef={ctrl_coef:.4f}, OR={ctrl_or:.4f}, p={ctrl_p:.6g}")

# Also compute 95% CI for controlled odds ratio
conf_int = model_ctrl.conf_int().loc['female']
ctrl_or_ci = np.exp(conf_int)
print(f"Controlled OR 95% CI: [{ctrl_or_ci[0]:.4f}, {ctrl_or_ci[1]:.4f}]")

# Save a small JSON summary to file for easier parsing if needed
summary = {
    'n': int(len(df)),
    'accept_rate_male': float(rate_by_gender.loc[0]),
    'accept_rate_female': float(rate_by_gender.loc[1]),
    'rate_diff_female_minus_male': float(rate_diff),
    'chi2_p': float(p),
    'simple_logit_coef_female': float(simple_coef),
    'simple_logit_or_female': float(simple_or),
    'simple_logit_p': float(simple_p),
    'ctrl_logit_coef_female': float(ctrl_coef),
    'ctrl_logit_or_female': float(ctrl_or),
    'ctrl_logit_p': float(ctrl_p),
    'ctrl_logit_or_ci_low': float(ctrl_or_ci[0]),
    'ctrl_logit_or_ci_high': float(ctrl_or_ci[1])
}

import json
with open('analysis_summary.json','w') as f:
    json.dump(summary,f,indent=2)
