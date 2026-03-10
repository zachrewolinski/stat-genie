import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv('crofoot.csv')

# Map variables based on info.json descriptions
# Outcome: m_focal = 1 if focal won contest
outcome = 'm_focal'

# Group sizes: f_other = number of individuals in focal group
# win = number of individuals in other group
# Relative group size (focal - other)
df['rel_group_size'] = df['f_other'] - df['win']

# Contest location: m_other = distance of focal group from its home range center
# n_focal = distance of other group from its home range center
# Relative location: positive when contest is closer to focal group center
# (other group's distance minus focal group's distance)
df['rel_location'] = df['n_focal'] - df['m_other']

# Standardize predictors for comparability
for col in ['rel_group_size', 'rel_location']:
    df[f'z_{col}'] = (df[col] - df[col].mean()) / df[col].std(ddof=0)

# Prepare data for logistic regression
X = df[['z_rel_group_size', 'z_rel_location']]
X = sm.add_constant(X)
y = df[outcome]

# Fit logistic regression using GLM for easier robust covariance handling
model = sm.GLM(y, X, family=sm.families.Binomial())
result = model.fit()

# Cluster-robust standard errors by dyad (if available)
try:
    clustered = model.fit(cov_type='cluster', cov_kwds={'groups': df['dyad']})
except Exception:
    clustered = None

# Also fit simple univariate models for each predictor
uni_results = {}
for col in ['z_rel_group_size', 'z_rel_location']:
    X_uni = sm.add_constant(df[[col]])
    res_uni = sm.GLM(y, X_uni, family=sm.families.Binomial()).fit()
    uni_results[col] = res_uni

print('N:', len(df))
print('\nMain model (cluster-robust by dyad):')
if clustered is not None:
    print(clustered.summary())
else:
    print('Cluster-robust estimation failed; using default covariance.')
    print(result.summary())

print('\nUnivariate models:')
for col, res in uni_results.items():
    print(f"\nPredictor: {col}")
    print(res.summary())

# Save key stats for reporting
use_res = clustered if clustered is not None else result
summary = {
    'n': len(df),
    'main_params': use_res.params.to_dict(),
    'main_pvalues': use_res.pvalues.to_dict(),
    'main_bse': use_res.bse.to_dict(),
    'main_or': (np.exp(use_res.params)).to_dict(),
}

pd.Series(summary).to_json('analysis_summary.json')
