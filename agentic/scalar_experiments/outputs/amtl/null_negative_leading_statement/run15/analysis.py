import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv('amtl.csv')

# Prepare data
# Ensure valid sockets and num_amtl
_df = _df.copy()
_df = _df.dropna(subset=['num_amtl', 'sockets', 'age', 'prob_male', 'tooth_class', 'genus'])
_df = _df[_df['sockets'] > 0]
_df = _df[_df['num_amtl'].between(0, _df['sockets'])]

# Binary indicator for Homo sapiens
_df['is_homo'] = (_df['genus'] == 'Homo sapiens').astype(int)

# Categorical tooth_class
_tooth = pd.get_dummies(_df['tooth_class'], prefix='tooth', drop_first=True)

# Design matrix
X = pd.concat([
    pd.Series(1.0, index=_df.index, name='intercept'),
    _df[['is_homo', 'age', 'prob_male']].astype(float),
    _tooth.astype(float),
], axis=1)

# Binomial endog with successes/failures
successes = _df['num_amtl'].astype(float)
failures = (_df['sockets'] - _df['num_amtl']).astype(float)
endog = np.column_stack([successes, failures])

model = sm.GLM(endog, X, family=sm.families.Binomial())
result = model.fit()

coef = result.params['is_homo']
se = result.bse['is_homo']

# Wald 95% CI
z = 1.96
ci_low = coef - z * se
ci_high = coef + z * se

# Convert to odds ratio
or_ = float(np.exp(coef))
or_low = float(np.exp(ci_low))
or_high = float(np.exp(ci_high))

# Predicted marginal effect: difference in predicted AMTL rate at mean covariates
mean_row = X.mean()
# Two scenarios: is_homo 0 and 1
mean_non = mean_row.copy()
mean_non['is_homo'] = 0.0
mean_homo = mean_row.copy()
mean_homo['is_homo'] = 1.0

pred_non = result.predict(mean_non)
pred_homo = result.predict(mean_homo)

output = {
    'n': int(_df.shape[0]),
    'coef_is_homo_logodds': float(coef),
    'se_is_homo': float(se),
    'ci_logodds_low': float(ci_low),
    'ci_logodds_high': float(ci_high),
    'odds_ratio': or_,
    'odds_ratio_ci_low': or_low,
    'odds_ratio_ci_high': or_high,
    'pred_rate_non_homo': float(pred_non),
    'pred_rate_homo': float(pred_homo),
    'pred_rate_diff': float(pred_homo - pred_non),
}

print(output)
