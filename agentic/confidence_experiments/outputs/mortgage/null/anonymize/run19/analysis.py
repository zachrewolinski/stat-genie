import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
_df = pd.read_csv('mortgage.csv')

# Define key columns
female_col = 'feature2'  # 1 if female
accept_col = 'feature14'  # 1 if accepted

# Basic sanity checks
# Drop rows with missing key values
_df = _df.dropna(subset=[female_col, accept_col]).copy()

# Basic counts
n_total = len(_df)
counts = _df[female_col].value_counts().sort_index()

# Approval rates by gender
approval_rates = _df.groupby(female_col)[accept_col].mean()

# 2x2 contingency for chi-square
contingency = pd.crosstab(_df[female_col], _df[accept_col])
# Ensure order: rows [male=0, female=1], cols [denied=0, accepted=1]
contingency = contingency.reindex(index=[0,1], columns=[0,1])
chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency)

# Logistic regression with controls
# Exclude feature1 (unique id-like), feature11 (denied) to avoid collinearity with accept
control_cols = [
    'feature2',  # female
    'feature3',  # black
    'feature4',  # housing expense ratio
    'feature5',  # self-employed
    'feature6',  # married
    'feature7',  # mortgage credit score
    'feature8',  # consumer credit score
    'feature9',  # bad credit
    'feature10', # debt ratio
    'feature12', # loan-to-value
    'feature13', # PMI denied
]

reg_df = _df.dropna(subset=control_cols + [accept_col]).copy()
X = reg_df[control_cols]
y = reg_df[accept_col]
X = sm.add_constant(X, has_constant='add')

# Use GLM binomial with robust SEs for stability
model = sm.GLM(y, X, family=sm.families.Binomial())
res = model.fit(cov_type='HC1')

coef_female = res.params['feature2']
se_female = res.bse['feature2']
p_female = res.pvalues['feature2']

# Odds ratio and 95% CI for female coefficient
odds_ratio = float(np.exp(coef_female))
ci_low, ci_high = res.conf_int().loc['feature2']
ci_low_or = float(np.exp(ci_low))
ci_high_or = float(np.exp(ci_high))

# Compute raw difference in approval rates
rate_male = float(approval_rates.loc[0]) if 0 in approval_rates.index else float('nan')
rate_female = float(approval_rates.loc[1]) if 1 in approval_rates.index else float('nan')
rate_diff = rate_female - rate_male if (not np.isnan(rate_female) and not np.isnan(rate_male)) else float('nan')

# Determine response scale
# Base on significance and effect size
alpha = 0.05
if p_female < alpha:
    # significant; direction matters
    # Scale strength by absolute log-odds effect; moderate -> around 65-75
    abs_effect = abs(coef_female)
    if abs_effect < 0.1:
        response = 60
    elif abs_effect < 0.25:
        response = 70
    elif abs_effect < 0.5:
        response = 80
    else:
        response = 90
else:
    # not significant -> lean toward No
    # If raw difference is tiny, closer to strong No
    abs_diff = abs(rate_diff) if not np.isnan(rate_diff) else 0
    if abs_diff < 0.01:
        response = 35
    elif abs_diff < 0.03:
        response = 40
    elif abs_diff < 0.05:
        response = 45
    else:
        response = 50

# Build explanation
explanation = (
    f"Analyzed {n_total} applications. Approval rate for males (female=0) was "
    f"{rate_male:.3f} and for females (female=1) was {rate_female:.3f}, a difference of {rate_diff:.3f}. "
    f"A chi-square test of independence between gender and approval yielded p={p_chi2:.4g}. "
    f"A logistic regression controlling for race, credit scores, debt ratios, loan-to-value, "
    f"marital/self-employment status, bad credit history, and PMI denial found the female coefficient "
    f"{coef_female:.3f} (SE {se_female:.3f}), odds ratio {odds_ratio:.3f} with 95% CI "
    f"[{ci_low_or:.3f}, {ci_high_or:.3f}], p={p_female:.4g}. "
)

if p_female < alpha:
    explanation += (
        "This indicates a statistically significant association between gender and approval, "
        "so the answer leans Yes."
    )
else:
    explanation += (
        "This does not provide statistically significant evidence that gender affects approval after "
        "accounting for observed credit-related factors, so the answer leans No."
    )

# Write conclusion
output = {
    "response": int(response),
    "explanation": explanation
}

with open('conclusion.txt', 'w', encoding='utf-8') as f:
    json.dump(output, f)
