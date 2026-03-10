import json
import math
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest

DATA_PATH = 'mortgage.csv'

df = pd.read_csv(DATA_PATH)

# Define outcome and predictors
outcome = 'accept'
# Avoid using 'deny' since it's the complement of accept
predictors = [
    'female',
    'black',
    'housing_expense_ratio',
    'self_employed',
    'married',
    'mortgage_credit',
    'consumer_credit',
    'bad_history',
    'PI_ratio',
    'loan_to_value',
]

needed_cols = [outcome] + predictors

# Drop rows with missing data in relevant columns
model_df = df[needed_cols].dropna().copy()

n_total = len(df)
n_model = len(model_df)

# Unadjusted approval rates by gender
female_mask = model_df['female'] == 1
male_mask = model_df['female'] == 0

female_n = int(female_mask.sum())
male_n = int(male_mask.sum())

female_accept_rate = model_df.loc[female_mask, outcome].mean()
male_accept_rate = model_df.loc[male_mask, outcome].mean()

# Two-proportion z-test for difference in approval rates
successes = np.array([
    model_df.loc[female_mask, outcome].sum(),
    model_df.loc[male_mask, outcome].sum()
])
counts = np.array([female_n, male_n])

# Guard against edge cases
if female_n > 0 and male_n > 0:
    z_stat, p_value_unadj = proportions_ztest(successes, counts)
else:
    z_stat, p_value_unadj = np.nan, np.nan

# Logistic regression via GLM (adjusted)
X = model_df[predictors]
X = sm.add_constant(X, has_constant='add')
y = model_df[outcome]

glm_model = sm.GLM(y, X, family=sm.families.Binomial())
result = glm_model.fit()

# Robust covariance (HC1)
if hasattr(result, 'get_robustcov_results'):
    robust_result = result.get_robustcov_results(cov_type='HC1')
else:
    robust_result = result

# Extract female effect
coef_female = robust_result.params['female']
se_female = robust_result.bse['female']
pval_female = robust_result.pvalues['female']
ci_low, ci_high = robust_result.conf_int().loc['female']

odds_ratio = math.exp(coef_female)
ci_or_low = math.exp(ci_low)
ci_or_high = math.exp(ci_high)

# Predicted probability difference at mean covariates
mean_cov = X.mean()
mean_cov_f = mean_cov.copy()
mean_cov_f['female'] = 1
mean_cov_m = mean_cov.copy()
mean_cov_m['female'] = 0

pred_f = result.predict(mean_cov_f)[0]
pred_m = result.predict(mean_cov_m)[0]

prob_diff = pred_f - pred_m

summary = {
    'n_total': n_total,
    'n_model': n_model,
    'female_n': female_n,
    'male_n': male_n,
    'female_accept_rate': float(female_accept_rate),
    'male_accept_rate': float(male_accept_rate),
    'unadjusted_rate_diff': float(female_accept_rate - male_accept_rate),
    'unadjusted_p_value': float(p_value_unadj),
    'logit_coef_female': float(coef_female),
    'logit_se_female': float(se_female),
    'logit_p_value_female': float(pval_female),
    'logit_or_female': float(odds_ratio),
    'logit_or_ci_low': float(ci_or_low),
    'logit_or_ci_high': float(ci_or_high),
    'pred_prob_female_mean_cov': float(pred_f),
    'pred_prob_male_mean_cov': float(pred_m),
    'pred_prob_diff_mean_cov': float(prob_diff),
}

with open('analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
