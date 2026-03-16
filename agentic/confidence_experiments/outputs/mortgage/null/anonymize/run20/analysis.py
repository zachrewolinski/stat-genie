import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

# Load data
path = 'mortgage.csv'
df = pd.read_csv(path)

# Define columns
outcome = 'feature14'  # 1 accepted, 0 denied
female = 'feature2'    # 1 female, 0 male

# Basic cleanup
# Drop rows with missing in columns used
predictors = [c for c in df.columns if c not in ['feature11', 'feature14']]
cols_needed = [outcome] + predictors
clean = df[cols_needed].dropna().copy()

# Ensure binary outcomes are 0/1
clean[outcome] = clean[outcome].astype(int)

# Acceptance rates by gender
rates = clean.groupby(female)[outcome].mean()
counts = clean.groupby(female)[outcome].agg(['sum','count'])

# 2x2 contingency table for chi-square
# rows: female=0,1 ; cols: accepted=1, denied=0
contingency = pd.crosstab(clean[female], clean[outcome])
chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency)

# Effect sizes
acc_male = rates.loc[0.0] if 0.0 in rates.index else rates.loc[0]
acc_female = rates.loc[1.0] if 1.0 in rates.index else rates.loc[1]

# Odds ratio from 2x2
# add 0.5 to avoid division by zero in rare case
male_accept = contingency.loc[0,1]
male_deny = contingency.loc[0,0]
fem_accept = contingency.loc[1,1]
fem_deny = contingency.loc[1,0]

or_unadj = (fem_accept / fem_deny) / (male_accept / male_deny)

# Unadjusted logistic regression
X_unadj = sm.add_constant(clean[[female]])
model_unadj = sm.Logit(clean[outcome], X_unadj).fit(disp=0)

# Adjusted logistic regression with all predictors (excluding outcome)
X_adj = sm.add_constant(clean[predictors])
model_adj = sm.Logit(clean[outcome], X_adj).fit(disp=0)

coef_female = model_adj.params[female]
se_female = model_adj.bse[female]
p_female = model_adj.pvalues[female]

or_adj = np.exp(coef_female)

# Summarize
summary = {
    "n_total": int(clean.shape[0]),
    "acceptance_rate_male": float(acc_male),
    "acceptance_rate_female": float(acc_female),
    "acceptance_rate_diff_female_minus_male": float(acc_female - acc_male),
    "chi2_p_value": float(p_chi2),
    "odds_ratio_unadjusted": float(or_unadj),
    "logit_unadjusted_p": float(model_unadj.pvalues[female]),
    "logit_adjusted_coef": float(coef_female),
    "logit_adjusted_or": float(or_adj),
    "logit_adjusted_p": float(p_female),
}

print(json.dumps(summary, indent=2))
