import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'crofoot.csv'
df = pd.read_csv(path)

# Create predictors
# Relative group size (focal - other) and location advantage (other distance - focal distance)
df['size_diff'] = df['n_focal'] - df['n_other']
# Positive loc_adv means focal is closer to its home range center than the other group
# (other further from its own center), which is a plausible location advantage for focal.
df['loc_adv'] = df['dist_other'] - df['dist_focal']

# Standardize for comparability
for col in ['size_diff', 'loc_adv']:
    df[col + '_z'] = (df[col] - df[col].mean()) / df[col].std(ddof=0)

# Logistic regression with both predictors
model = smf.glm('win ~ size_diff_z + loc_adv_z', data=df, family=sm.families.Binomial()).fit()

# Single-predictor models for context
model_size = smf.glm('win ~ size_diff_z', data=df, family=sm.families.Binomial()).fit()
model_loc = smf.glm('win ~ loc_adv_z', data=df, family=sm.families.Binomial()).fit()

# Odds ratios and confidence intervals
params = model.params
conf = model.conf_int()

odds_ratios = np.exp(params)
conf_or = np.exp(conf)

# Collect results
results = {
    'n': len(df),
    'model_params': params.to_dict(),
    'model_pvalues': model.pvalues.to_dict(),
    'model_odds_ratios': odds_ratios.to_dict(),
    'model_or_ci_lower': conf_or[0].to_dict(),
    'model_or_ci_upper': conf_or[1].to_dict(),
    'model_size_pvalue': model_size.pvalues.to_dict(),
    'model_loc_pvalue': model_loc.pvalues.to_dict(),
}

print(json.dumps(results, indent=2))
