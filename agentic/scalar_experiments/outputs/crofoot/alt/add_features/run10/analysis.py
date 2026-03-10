import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data
DF = pd.read_csv('crofoot.csv')

# Focus on relevant variables
# Relative group size: focal minus other
DF['rel_size'] = DF['n_focal'] - DF['n_other']
# Relative contest location: positive means contest is closer to focal group's center
DF['rel_dist'] = DF['dist_other'] - DF['dist_focal']

# Prepare design matrix
X = DF[['rel_size', 'rel_dist']].copy()
X = sm.add_constant(X)
y = DF['win']

# Fit logistic regression (GLM Binomial)
model = sm.GLM(y, X, family=sm.families.Binomial())
result = model.fit()

# Also check alternative size metric: log ratio (add small epsilon to avoid divide by zero)
DF['log_size_ratio'] = np.log((DF['n_focal'] + 0.5) / (DF['n_other'] + 0.5))
X2 = sm.add_constant(DF[['log_size_ratio', 'rel_dist']])
model2 = sm.GLM(y, X2, family=sm.families.Binomial())
result2 = model2.fit()

# Save key outputs for inspection
summary = {
    'n': int(len(DF)),
    'model_rel_size': {
        'coef': result.params.to_dict(),
        'pvalues': result.pvalues.to_dict(),
        'aic': float(result.aic),
    },
    'model_log_size_ratio': {
        'coef': result2.params.to_dict(),
        'pvalues': result2.pvalues.to_dict(),
        'aic': float(result2.aic),
    },
}

with open('analysis_outputs.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(result.summary())
print('\n---\n')
print(result2.summary())
