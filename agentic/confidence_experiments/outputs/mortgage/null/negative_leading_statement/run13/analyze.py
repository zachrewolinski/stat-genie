import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data
csv_path = 'mortgage.csv'
df = pd.read_csv(csv_path)

# Basic cleaning: drop unnamed index column if present
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Ensure binary variables are numeric 0/1
binary_cols = ['female','black','self_employed','married','bad_history','deny','denied_PMI','accept']
for c in binary_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# Create approval outcome: accept (1) vs deny (0)
# accept already provided; if missing use 1 - deny
if 'accept' not in df.columns and 'deny' in df.columns:
    df['accept'] = 1 - df['deny']

# Drop rows with missing in key variables
key_cols = ['female','accept']
analysis_df = df.dropna(subset=key_cols).copy()

# Descriptive rates
rate_by_gender = analysis_df.groupby('female')['accept'].agg(['mean','count'])

# Two-proportion z-test (female vs male) for accept rate
# male=0, female=1
male = analysis_df[analysis_df['female'] == 0]['accept']
female = analysis_df[analysis_df['female'] == 1]['accept']
count = np.array([female.sum(), male.sum()])
obs = np.array([female.count(), male.count()])
# Two-proportion z-test using statsmodels
from statsmodels.stats.proportion import proportions_ztest
z_stat, p_val = proportions_ztest(count, obs)

# Chi-square test of independence
contingency = pd.crosstab(analysis_df['female'], analysis_df['accept'])
chi2, chi_p, dof, expected = stats.chi2_contingency(contingency)

# Logistic regression without controls
logit_simple = smf.logit('accept ~ female', data=analysis_df).fit(disp=0)

# Logistic regression with controls (credit-related and applicant factors)
control_cols = ['black','housing_expense_ratio','self_employed','married','mortgage_credit',
                'consumer_credit','bad_history','PI_ratio','loan_to_value','denied_PMI']
# Keep only controls that exist and have variation
controls = [c for c in control_cols if c in analysis_df.columns]
# Drop missing for regression
reg_df = analysis_df.dropna(subset=['accept','female'] + controls).copy()
# Build formula
formula = 'accept ~ female'
if controls:
    formula += ' + ' + ' + '.join(controls)
logit_controls = smf.logit(formula, data=reg_df).fit(disp=0)

# Effect size: odds ratio for female
or_simple = np.exp(logit_simple.params['female'])
or_controls = np.exp(logit_controls.params['female'])

results = {
    'n_total': int(len(analysis_df)),
    'rate_by_gender': {
        'male_accept_rate': float(rate_by_gender.loc[0,'mean']) if 0 in rate_by_gender.index else None,
        'female_accept_rate': float(rate_by_gender.loc[1,'mean']) if 1 in rate_by_gender.index else None,
        'male_n': int(rate_by_gender.loc[0,'count']) if 0 in rate_by_gender.index else None,
        'female_n': int(rate_by_gender.loc[1,'count']) if 1 in rate_by_gender.index else None,
    },
    'two_prop_ztest': {'z': float(z_stat), 'p': float(p_val)},
    'chi_square': {'chi2': float(chi2), 'p': float(chi_p), 'dof': int(dof)},
    'logit_simple': {
        'coef_female': float(logit_simple.params['female']),
        'p_female': float(logit_simple.pvalues['female']),
        'odds_ratio_female': float(or_simple)
    },
    'logit_controls': {
        'coef_female': float(logit_controls.params['female']),
        'p_female': float(logit_controls.pvalues['female']),
        'odds_ratio_female': float(or_controls)
    },
    'formula_controls': formula,
    'n_reg': int(len(reg_df))
}

print(json.dumps(results, indent=2))
