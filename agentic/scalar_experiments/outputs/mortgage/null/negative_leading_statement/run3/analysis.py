import json
import numpy as np
import pandas as pd
import statsmodels.api as sm


df = pd.read_csv('mortgage.csv')

# Drop unnamed index column if present
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Basic checks
required_cols = ['female', 'accept', 'deny']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# Ensure numeric, allow missing
for col in ['female', 'accept', 'deny']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Bivariate analysis: acceptance rates by gender
# Use complete cases for bivariate analysis
df_bi = df[['female', 'accept']].dropna()
accept_rates = df_bi.groupby('female')['accept'].mean()
counts = df_bi['female'].value_counts().sort_index()

# Two-proportion z-test (female vs male) on acceptance
# female==1, male==0
succ_f = df_bi.loc[df_bi['female'] == 1, 'accept'].sum()
obs_f = (df_bi['female'] == 1).sum()
succ_m = df_bi.loc[df_bi['female'] == 0, 'accept'].sum()
obs_m = (df_bi['female'] == 0).sum()
count = [succ_f, succ_m]
obs = [obs_f, obs_m]
stat, pval_prop = sm.stats.proportions_ztest(count, obs)

# Multivariate logistic regression
# Outcome: accept (1=accepted)
# Predictors: female + controls
controls = [
    'black',
    'housing_expense_ratio',
    'self_employed',
    'married',
    'mortgage_credit',
    'consumer_credit',
    'bad_history',
    'PI_ratio',
    'loan_to_value',
    'denied_PMI'
]

X = df[['female'] + controls].copy()
X = sm.add_constant(X, has_constant='add')

y = df['accept']

glm_model = sm.GLM(y, X, family=sm.families.Binomial(), missing='drop')
result = glm_model.fit()

# Robust SEs (HC1) via internal helper (public API missing for GLMResults)
result._get_robustcov_results(cov_type='HC1')

female_coef = float(result.params['female'])
female_se = float(result.bse['female'])
female_p = float(result.pvalues['female'])

# Odds ratio and CI
odds_ratio = float(np.exp(female_coef))
ci_low = float(np.exp(female_coef - 1.96 * female_se))
ci_high = float(np.exp(female_coef + 1.96 * female_se))

# Save results for report
results = {
    'n': int(len(df)),
    'accept_rate_female': float(accept_rates.loc[1]) if 1 in accept_rates.index else None,
    'accept_rate_male': float(accept_rates.loc[0]) if 0 in accept_rates.index else None,
    'count_female': int(obs_f),
    'count_male': int(obs_m),
    'prop_test_p': float(pval_prop),
    'female_logit_coef': float(female_coef),
    'female_logit_p': float(female_p),
    'female_odds_ratio': odds_ratio,
    'female_or_ci_low': ci_low,
    'female_or_ci_high': ci_high,
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
