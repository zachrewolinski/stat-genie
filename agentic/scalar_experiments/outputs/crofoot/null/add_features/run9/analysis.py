import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'crofoot.csv'
df = pd.read_csv(path)

# Focus on variables relevant to research question
needed = ['win', 'n_focal', 'n_other', 'dist_focal', 'dist_other']
missing = [c for c in needed if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Create relative group size and contest location metrics
# Relative group size: focal minus other (positive means focal larger)
# Contest location: other distance to its center minus focal distance to its center
# Positive means contest closer to focal home range center than to the other group's center
analysis_df = df[needed].copy()
analysis_df['size_diff'] = analysis_df['n_focal'] - analysis_df['n_other']
analysis_df['loc_diff'] = analysis_df['dist_other'] - analysis_df['dist_focal']

# Basic summaries
summary = {
    'n': int(len(analysis_df)),
    'win_rate': float(analysis_df['win'].mean()),
    'size_diff_mean': float(analysis_df['size_diff'].mean()),
    'loc_diff_mean': float(analysis_df['loc_diff'].mean()),
}

# Point-biserial correlations
corr_size = analysis_df['win'].corr(analysis_df['size_diff'])
corr_loc = analysis_df['win'].corr(analysis_df['loc_diff'])

# Logistic regression
# Use GLM binomial for stable inference
model = smf.glm('win ~ size_diff + loc_diff', data=analysis_df, family=sm.families.Binomial()).fit()

# Extract coefficients, p-values, odds ratios
params = model.params
pvalues = model.pvalues
conf = model.conf_int()

odds = np.exp(params)
conf_odds = np.exp(conf)

results = {
    'summary': summary,
    'corr_size': float(corr_size),
    'corr_loc': float(corr_loc),
    'coef': params.to_dict(),
    'pvalues': pvalues.to_dict(),
    'odds_ratios': odds.to_dict(),
    'odds_ci_lower': conf_odds[0].to_dict(),
    'odds_ci_upper': conf_odds[1].to_dict(),
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(model.summary())
print('\nResults saved to analysis_results.json')
