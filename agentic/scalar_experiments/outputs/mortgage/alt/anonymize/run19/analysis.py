import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data
csv_path = 'mortgage.csv'
df = pd.read_csv(csv_path)

# Column names
female_col = 'feature2'  # 1 if female
accepted_col = 'feature14'  # 1 if accepted

# Basic sanity
# Drop rows with missing in key columns
analysis_df = df[[female_col, accepted_col]].dropna()

# Contingency table
ct = pd.crosstab(analysis_df[female_col], analysis_df[accepted_col])
# Ensure ordering: rows 0 male,1 female; cols 0 denied,1 accepted
ct = ct.reindex(index=[0,1], columns=[0,1], fill_value=0)

# Rates
male_accept_rate = ct.loc[0,1] / ct.loc[0].sum() if ct.loc[0].sum() > 0 else np.nan
female_accept_rate = ct.loc[1,1] / ct.loc[1].sum() if ct.loc[1].sum() > 0 else np.nan
rate_diff = female_accept_rate - male_accept_rate

# Chi-square test of independence
chi2, p_chi, dof, expected = stats.chi2_contingency(ct)

# Effect size: odds ratio
# Add 0.5 continuity correction if any zero
ct_or = ct.copy().astype(float)
if (ct_or == 0).any().any():
    ct_or += 0.5
odds_male = ct_or.loc[0,1] / ct_or.loc[0,0]
odds_female = ct_or.loc[1,1] / ct_or.loc[1,0]
odds_ratio = odds_female / odds_male

# Adjusted logistic regression
# Use a set of controls from metadata
control_cols = [
    'feature3',  # Black
    'feature4',  # housing expense ratio
    'feature5',  # self-employed
    'feature6',  # married
    'feature7',  # mortgage credit score
    'feature8',  # consumer credit score
    'feature9',  # bad credit history
    'feature10', # total debt payments to income ratio
    'feature12', # loan-to-value ratio
    'feature13', # denied PMI
]

model_cols = [female_col, accepted_col] + control_cols
model_df = df[model_cols].dropna()

X = model_df[[female_col] + control_cols]
X = sm.add_constant(X, has_constant='add')
y = model_df[accepted_col]

logit_model = sm.Logit(y, X)
try:
    result = logit_model.fit(disp=False)
    coef = result.params[female_col]
    se = result.bse[female_col]
    p_logit = result.pvalues[female_col]
    # odds ratio for female effect
    adj_or = np.exp(coef)
except Exception as e:
    result = None
    coef = np.nan
    se = np.nan
    p_logit = np.nan
    adj_or = np.nan

output = {
    'n_total': len(df),
    'n_analysis': len(analysis_df),
    'contingency_table': ct.to_dict(),
    'male_accept_rate': male_accept_rate,
    'female_accept_rate': female_accept_rate,
    'accept_rate_diff_female_minus_male': rate_diff,
    'chi2': chi2,
    'chi2_p_value': p_chi,
    'odds_ratio_female_vs_male_unadjusted': odds_ratio,
    'logit_female_coef': coef,
    'logit_female_se': se,
    'logit_female_p_value': p_logit,
    'logit_female_adj_odds_ratio': adj_or,
    'n_logit': len(model_df),
}

print(json.dumps(output, indent=2))
