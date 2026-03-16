import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
DATA_PATH = 'mortgage.csv'
df = pd.read_csv(DATA_PATH)

# Basic cleaning: drop rows with missing in needed columns
cols = [
    'accept', 'deny', 'female', 'black', 'housing_expense_ratio', 'self_employed',
    'married', 'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio',
    'loan_to_value', 'denied_PMI'
]
missing_cols = [c for c in cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")

analysis_df = df[cols].copy()

# Ensure binary types are numeric 0/1
# Drop rows with any missing values in analysis columns
analysis_df = analysis_df.dropna()

# Unadjusted acceptance rates by gender
rate_by_gender = analysis_df.groupby('female')['accept'].mean()
count_by_gender = analysis_df.groupby('female')['accept'].count()

# Two-proportion z-test (equivalent to chi-square for 2x2)
# Contingency table: female (rows) x accept (cols)
contingency = pd.crosstab(analysis_df['female'], analysis_df['accept'])
chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency)

# Logistic regression with controls
X_cols = [
    'female', 'black', 'housing_expense_ratio', 'self_employed', 'married',
    'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio',
    'loan_to_value', 'denied_PMI'
]
X = analysis_df[X_cols]
X = sm.add_constant(X, has_constant='add')

y = analysis_df['accept']

# Use GLM Binomial for stable inference
model = sm.GLM(y, X, family=sm.families.Binomial())
result = model.fit()

# Extract female coefficient
coef = result.params['female']
se = result.bse['female']
p_value = result.pvalues['female']

# Odds ratio and 95% CI
or_female = float(np.exp(coef))
ci_low = float(np.exp(coef - 1.96 * se))
ci_high = float(np.exp(coef + 1.96 * se))

# Marginal effect approximation at mean (for interpretability)
# Compute predicted probability difference when female changes from 0 to 1 at mean covariates
X_mean = X.mean().to_frame().T
X_mean_f0 = X_mean.copy(); X_mean_f0['female'] = 0
X_mean_f1 = X_mean.copy(); X_mean_f1['female'] = 1
pred_f0 = result.predict(X_mean_f0).iloc[0]
pred_f1 = result.predict(X_mean_f1).iloc[0]

# Prepare outputs for explanation
summary = {
    'n': int(len(analysis_df)),
    'accept_rate_male': float(rate_by_gender.get(0.0, rate_by_gender.get(0))),
    'accept_rate_female': float(rate_by_gender.get(1.0, rate_by_gender.get(1))),
    'count_male': int(count_by_gender.get(0.0, count_by_gender.get(0))),
    'count_female': int(count_by_gender.get(1.0, count_by_gender.get(1))),
    'chi2_p_value': float(p_chi2),
    'logit_coef_female': float(coef),
    'logit_p_value_female': float(p_value),
    'odds_ratio_female': or_female,
    'odds_ratio_ci_low': ci_low,
    'odds_ratio_ci_high': ci_high,
    'pred_accept_male_at_mean': float(pred_f0),
    'pred_accept_female_at_mean': float(pred_f1),
    'pred_diff_female_minus_male_at_mean': float(pred_f1 - pred_f0),
}

print(json.dumps(summary, indent=2))
