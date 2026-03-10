import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
df = pd.read_csv('mortgage.csv')

# Clean column names if needed

# Basic checks
n_rows = len(df)

# Ensure binary columns are numeric
binary_cols = ['female','black','self_employed','married','bad_history','deny','accept','denied_PMI']
for col in binary_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Outcome
if 'deny' in df.columns:
    y_full = df['deny']
else:
    y_full = 1 - df['accept']

# Unadjusted association: difference in denial rates by gender
ct = pd.crosstab(df['female'], y_full)
# Ensure ordering: female=0 male? actually female=1; female=0 male
# Compute rates
rate_female = y_full[df['female'] == 1].mean()
rate_male = y_full[df['female'] == 0].mean()
rate_diff = rate_female - rate_male

# Chi-square test of independence
chi2, p_chi, dof, exp = stats.chi2_contingency(ct)

# Logistic regression (unadjusted)
X_unadj = sm.add_constant(df[['female']])
data_unadj = pd.concat([y_full, X_unadj], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
y_unadj = data_unadj.iloc[:, 0]
X_unadj = data_unadj.iloc[:, 1:]
logit_unadj = sm.Logit(y_unadj, X_unadj).fit(disp=False)
unadj_n = len(y_unadj)

# Logistic regression (adjusted) with available controls
control_cols = [
    'female','black','housing_expense_ratio','self_employed','married',
    'mortgage_credit','consumer_credit','bad_history','PI_ratio','loan_to_value','denied_PMI'
]
# Keep only columns that exist
control_cols = [c for c in control_cols if c in df.columns]
X_adj = sm.add_constant(df[control_cols])
data_adj = pd.concat([y_full, X_adj], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
y_adj = data_adj.iloc[:, 0]
X_adj = data_adj.iloc[:, 1:]
logit_adj = sm.Logit(y_adj, X_adj).fit(disp=False)
adj_n = len(y_adj)

# Alternative adjusted model without denied_PMI (potentially downstream of approval)
control_cols_no_pmi = [
    'female','black','housing_expense_ratio','self_employed','married',
    'mortgage_credit','consumer_credit','bad_history','PI_ratio','loan_to_value'
]
control_cols_no_pmi = [c for c in control_cols_no_pmi if c in df.columns]
X_adj2 = sm.add_constant(df[control_cols_no_pmi])
data_adj2 = pd.concat([y_full, X_adj2], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
y_adj2 = data_adj2.iloc[:, 0]
X_adj2 = data_adj2.iloc[:, 1:]
logit_adj2 = sm.Logit(y_adj2, X_adj2).fit(disp=False)
adj2_n = len(y_adj2)

# Extract female effect
female_coef_unadj = logit_unadj.params['female']
female_se_unadj = logit_unadj.bse['female']
female_p_unadj = logit_unadj.pvalues['female']

female_coef_adj = logit_adj.params['female']
female_se_adj = logit_adj.bse['female']
female_p_adj = logit_adj.pvalues['female']

female_coef_adj2 = logit_adj2.params['female']
female_se_adj2 = logit_adj2.bse['female']
female_p_adj2 = logit_adj2.pvalues['female']

# Odds ratios and CI

def odds_ratio_ci(coef, se, alpha=0.05):
    z = stats.norm.ppf(1 - alpha/2)
    lower = coef - z*se
    upper = coef + z*se
    return np.exp(coef), np.exp(lower), np.exp(upper)

or_unadj, or_unadj_l, or_unadj_u = odds_ratio_ci(female_coef_unadj, female_se_unadj)
or_adj, or_adj_l, or_adj_u = odds_ratio_ci(female_coef_adj, female_se_adj)
or_adj2, or_adj2_l, or_adj2_u = odds_ratio_ci(female_coef_adj2, female_se_adj2)

# Model fit summary info
pseudo_r2_adj = logit_adj.prsquared
pseudo_r2_adj2 = logit_adj2.prsquared

# Prepare results
results = {
    'n_rows': n_rows,
    'n_unadjusted_model': int(unadj_n),
    'n_adjusted_model': int(adj_n),
    'n_adjusted_model_no_pmi': int(adj2_n),
    'denial_rate_female': rate_female,
    'denial_rate_male': rate_male,
    'rate_diff_female_minus_male': rate_diff,
    'chi2_pvalue': p_chi,
    'logit_unadj_female_coef': female_coef_unadj,
    'logit_unadj_female_pvalue': female_p_unadj,
    'logit_unadj_or': or_unadj,
    'logit_unadj_or_ci': [or_unadj_l, or_unadj_u],
    'logit_adj_female_coef': female_coef_adj,
    'logit_adj_female_pvalue': female_p_adj,
    'logit_adj_or': or_adj,
    'logit_adj_or_ci': [or_adj_l, or_adj_u],
    'logit_adj_no_pmi_female_coef': female_coef_adj2,
    'logit_adj_no_pmi_female_pvalue': female_p_adj2,
    'logit_adj_no_pmi_or': or_adj2,
    'logit_adj_no_pmi_or_ci': [or_adj2_l, or_adj2_u],
    'pseudo_r2_adj': pseudo_r2_adj,
    'pseudo_r2_adj_no_pmi': pseudo_r2_adj2,
}

with open('analysis_results.json','w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
