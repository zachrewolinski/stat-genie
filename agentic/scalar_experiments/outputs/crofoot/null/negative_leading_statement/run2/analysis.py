import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data
csv_path = 'crofoot.csv'
df = pd.read_csv(csv_path)

# Construct predictors
# Relative group size: focal size minus other size
# Contest location: positive when contest is closer to focal home-range center
# (other farther away than focal)
df['size_diff'] = df['n_focal'] - df['n_other']
df['dist_diff'] = df['dist_other'] - df['dist_focal']

# Scale distance for interpretability (per 100m)
df['dist_diff_100'] = df['dist_diff'] / 100.0

# Prepare model
X = df[['size_diff', 'dist_diff_100']]
X = sm.add_constant(X)
y = df['win']

# Fit logistic regression
model = sm.Logit(y, X)
result = model.fit(disp=False)

# Extract coefficients, p-values, and odds ratios
params = result.params
conf = result.conf_int()
pvalues = result.pvalues

odds_ratios = np.exp(params)
conf_or = np.exp(conf)

summary = {
    'n_obs': int(result.nobs),
    'coef': params.to_dict(),
    'pvalues': pvalues.to_dict(),
    'odds_ratios': odds_ratios.to_dict(),
    'or_ci_lower': conf_or[0].to_dict(),
    'or_ci_upper': conf_or[1].to_dict(),
    'pseudo_r2': result.prsquared,
}

# Also fit univariate models for robustness
models = {}
for col in ['size_diff', 'dist_diff_100']:
    X1 = sm.add_constant(df[[col]])
    res1 = sm.Logit(y, X1).fit(disp=False)
    models[col] = {
        'coef': res1.params.to_dict(),
        'pvalues': res1.pvalues.to_dict(),
        'odds_ratios': np.exp(res1.params).to_dict(),
        'or_ci_lower': np.exp(res1.conf_int()[0]).to_dict(),
        'or_ci_upper': np.exp(res1.conf_int()[1]).to_dict(),
        'pseudo_r2': res1.prsquared,
    }

out = {
    'multivariate': summary,
    'univariate': models,
}

with open('analysis_results.json', 'w') as f:
    json.dump(out, f, indent=2)

print(json.dumps(out, indent=2))
