import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest

# Load data

df = pd.read_csv('crofoot.csv')

# Derived variables
# Relative group size: focal - other (positive means focal larger)
# Relative location: dist_focal - dist_other (positive means contest closer to other group's center)
# Also compute absolute distances maybe.

df['rel_size'] = df['n_focal'] - df['n_other']
df['rel_dist'] = df['dist_focal'] - df['dist_other']

# Basic summaries
n = len(df)
win_rate = df['win'].mean()

# Simple contingency: win rates by size advantage sign
size_adv = pd.cut(df['rel_size'], bins=[-np.inf, -0.5, 0.5, np.inf], labels=['focal_smaller','equal','focal_larger'])
win_by_size = df.groupby(size_adv)['win'].agg(['mean','count'])

# Location advantage sign: contest closer to focal if dist_focal < dist_other (rel_dist negative)
loc_adv = pd.cut(df['rel_dist'], bins=[-np.inf, -1e-9, 1e-9, np.inf], labels=['closer_to_focal','equal','closer_to_other'])
win_by_loc = df.groupby(loc_adv)['win'].agg(['mean','count'])

# Logistic regression: win ~ rel_size + rel_dist
X = df[['rel_size','rel_dist']].copy()
X = sm.add_constant(X)
model = sm.GLM(df['win'], X, family=sm.families.Binomial())
res = model.fit()

# Also model with standardized predictors to compare effect sizes
Xz = df[['rel_size','rel_dist']].apply(lambda s: (s - s.mean())/s.std())
Xz = sm.add_constant(Xz)
res_z = sm.GLM(df['win'], Xz, family=sm.families.Binomial()).fit()

# Extract summary stats
params = res.params
conf = res.conf_int()
conf.columns = ['ci_low','ci_high']
summary = pd.concat([params, res.pvalues, conf], axis=1)
summary.columns = ['coef','pvalue','ci_low','ci_high']

summary_z = pd.concat([res_z.params, res_z.pvalues], axis=1)
summary_z.columns = ['coef_z','pvalue_z']

# Odds ratios
odds = np.exp(summary[['coef','ci_low','ci_high']])
odds.columns = ['odds_ratio','or_ci_low','or_ci_high']

# Save results to json for inspection
results = {
    'n': int(n),
    'win_rate': float(win_rate),
    'win_by_size': win_by_size.reset_index().to_dict(orient='records'),
    'win_by_loc': win_by_loc.reset_index().to_dict(orient='records'),
    'logit_summary': summary.reset_index().rename(columns={'index':'term'}).to_dict(orient='records'),
    'logit_odds': odds.reset_index().rename(columns={'index':'term'}).to_dict(orient='records'),
    'logit_z': summary_z.reset_index().rename(columns={'index':'term'}).to_dict(orient='records'),
    'model_aic': float(res.aic),
    'model_pseudo_r2': float(1 - res.deviance / res.null_deviance)
}

with open('analysis_results.json','w') as f:
    json.dump(results, f, indent=2)

print('done')
