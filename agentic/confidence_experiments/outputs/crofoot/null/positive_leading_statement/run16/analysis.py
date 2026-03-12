import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data
_df = pd.read_csv('crofoot.csv')

# Feature engineering
_df['size_diff'] = _df['n_focal'] - _df['n_other']
_df['location_diff'] = _df['dist_other'] - _df['dist_focal']  # positive => contest closer to focal center

# Logistic regression
X = _df[['size_diff', 'location_diff']]
X = sm.add_constant(X)
model = sm.GLM(_df['win'], X, family=sm.families.Binomial())
res = model.fit()

# Extract stats
params = res.params
pvals = res.pvalues
conf = res.conf_int()

# Compute odds ratios for interpretability
odds_ratios = np.exp(params)
conf_or = np.exp(conf)

summary = {
    'n': int(len(_df)),
    'coef': params.to_dict(),
    'pval': pvals.to_dict(),
    'odds_ratio': odds_ratios.to_dict(),
    'or_ci_low': conf_or[0].to_dict(),
    'or_ci_high': conf_or[1].to_dict(),
}

with open('analysis_results.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
