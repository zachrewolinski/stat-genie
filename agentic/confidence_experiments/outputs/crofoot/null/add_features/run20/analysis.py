import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data

df = pd.read_csv('crofoot.csv')

# Derived variables
# Relative group size: focal - other
# Relative location: focal distance from its center minus other group's distance from its center

df['size_diff'] = df['n_focal'] - df['n_other']
df['dist_diff'] = df['dist_focal'] - df['dist_other']

# Fit logistic regression models

models = {}

# Full model
models['full'] = smf.glm('win ~ size_diff + dist_diff', data=df, family=sm.families.Binomial()).fit()

# Size only
models['size_only'] = smf.glm('win ~ size_diff', data=df, family=sm.families.Binomial()).fit()

# Location only
models['loc_only'] = smf.glm('win ~ dist_diff', data=df, family=sm.families.Binomial()).fit()

# Interaction (optional)
models['interaction'] = smf.glm('win ~ size_diff * dist_diff', data=df, family=sm.families.Binomial()).fit()

# Summaries

for name, model in models.items():
    print(f"\n=== {name} ===")
    print(model.summary())

# Compute odds ratios and confidence intervals for full model
params = models['full'].params
conf = models['full'].conf_int()

or_table = pd.DataFrame({
    'coef': params,
    'odds_ratio': np.exp(params),
    'ci_low': np.exp(conf[0]),
    'ci_high': np.exp(conf[1]),
    'p_value': models['full'].pvalues,
})
print("\nFull model odds ratios:")
print(or_table)

# Marginal effects: probability change for typical values
# Estimate probability for size_diff +/- 1 and dist_diff +/- 100 (approx)

mean_size = df['size_diff'].mean()
mean_dist = df['dist_diff'].mean()

scenarios = pd.DataFrame({
    'size_diff': [mean_size-1, mean_size, mean_size+1, mean_size],
    'dist_diff': [mean_dist, mean_dist, mean_dist, mean_dist+100],
})
scenarios['pred_prob'] = models['full'].predict(scenarios)
print("\nScenario predictions (mean +/- changes):")
print(scenarios)

# Save key outputs for later use if needed
or_table.to_csv('analysis_odds_ratios.csv', index=True)
