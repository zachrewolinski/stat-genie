import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint

# Load data
path = 'crofoot.csv'
df = pd.read_csv(path)

# Define predictors
# Relative group size (focal - other)
df['rel_size'] = df['n_focal'] - df['n_other']
# Relative location: positive means contest closer to focal home range center
# (other distance minus focal distance)
df['rel_location'] = df['dist_other'] - df['dist_focal']

# Basic summaries
summary = {
    'n': len(df),
    'win_rate': df['win'].mean(),
    'rel_size_mean': df['rel_size'].mean(),
    'rel_size_sd': df['rel_size'].std(ddof=1),
    'rel_location_mean': df['rel_location'].mean(),
    'rel_location_sd': df['rel_location'].std(ddof=1),
}

# Correlations (point-biserial is Pearson with binary outcome)
corr_rel_size = df['win'].corr(df['rel_size'])
corr_rel_location = df['win'].corr(df['rel_location'])

# Logistic regression: win ~ rel_size + rel_location
X = df[['rel_size', 'rel_location']]
X = sm.add_constant(X)
model = sm.Logit(df['win'], X)
result = model.fit(disp=False)

# Logistic regression: win ~ rel_size (alone)
X1 = sm.add_constant(df[['rel_size']])
res1 = sm.Logit(df['win'], X1).fit(disp=False)

# Logistic regression: win ~ rel_location (alone)
X2 = sm.add_constant(df[['rel_location']])
res2 = sm.Logit(df['win'], X2).fit(disp=False)

# Odds ratios with 95% CI
params = result.params
conf = result.conf_int()
odds = np.exp(params)
odds_ci = np.exp(conf)

# Save key results
out = {
    'summary': summary,
    'corr_rel_size': corr_rel_size,
    'corr_rel_location': corr_rel_location,
    'logit_both': {
        'params': params.to_dict(),
        'pvalues': result.pvalues.to_dict(),
        'odds_ratios': odds.to_dict(),
        'odds_ci_low': odds_ci[0].to_dict(),
        'odds_ci_high': odds_ci[1].to_dict(),
        'pseudo_r2': result.prsquared,
    },
    'logit_rel_size': {
        'params': res1.params.to_dict(),
        'pvalues': res1.pvalues.to_dict(),
        'pseudo_r2': res1.prsquared,
    },
    'logit_rel_location': {
        'params': res2.params.to_dict(),
        'pvalues': res2.pvalues.to_dict(),
        'pseudo_r2': res2.prsquared,
    },
}

# Write results for inspection
pd.DataFrame({'metric': list(summary.keys()), 'value': list(summary.values())}).to_csv('analysis_summary.csv', index=False)
import json
with open('analysis_results.json', 'w') as f:
    json.dump(out, f, indent=2)

print('Done')
