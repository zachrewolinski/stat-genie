import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
path = 'panda_nuts.csv'
df = pd.read_csv(path)

# Rename columns for clarity
col_map = {
    'feature1': 'individual_id',
    'feature2': 'age',
    'feature3': 'sex',
    'feature4': 'hammer',
    'feature5': 'nuts_opened',
    'feature6': 'duration_sec',
    'feature7': 'helped'
}
df = df.rename(columns=col_map)

# Compute efficiency as nuts per second
# Avoid division by zero just in case
if (df['duration_sec'] <= 0).any():
    raise ValueError('Non-positive duration encountered')

df['efficiency'] = df['nuts_opened'] / df['duration_sec']

# Ensure categorical types
for col in ['sex', 'hammer', 'helped']:
    df[col] = df[col].astype('category')

# Fit OLS with cluster-robust SE by individual
formula = 'efficiency ~ age + C(sex) + C(helped) + C(hammer)'
model = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['individual_id']})

# Extract key stats
params = model.params
pvalues = model.pvalues
conf_int = model.conf_int()

# Prepare summary for variables of interest
results = {}
for term in ['age', 'C(sex)[T.m]', 'C(helped)[T.y]']:
    if term in params.index:
        results[term] = {
            'coef': float(params[term]),
            'pvalue': float(pvalues[term]),
            'ci_low': float(conf_int.loc[term, 0]),
            'ci_high': float(conf_int.loc[term, 1])
        }

# Model-level info
model_info = {
    'n_obs': int(model.nobs),
    'r2': float(model.rsquared),
    'adj_r2': float(model.rsquared_adj)
}

output = {
    'results': results,
    'model_info': model_info,
    'formula': formula
}

print(json.dumps(output, indent=2))
