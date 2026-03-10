import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data
info = json.load(open('info.json'))
df = pd.read_csv('crofoot.csv')

# Map columns based on info.json descriptions
outcome = 'm_focal'  # 1 if focal won contest
focal_size = 'f_other'  # number of individuals in focal group
other_size = 'win'      # number of individuals in other group
focal_dist = 'm_other'  # distance of focal group from center of its home range
other_dist = 'n_focal'  # distance of other group from center of its home range

# Derived predictors
# Relative group size: focal - other (positive means focal larger)
df['rel_size'] = df[focal_size] - df[other_size]
# Relative location: other_dist - focal_dist (positive means contest closer to focal home range)
df['rel_loc'] = df[other_dist] - df[focal_dist]

# Standardize predictors for comparability
for col in ['rel_size', 'rel_loc']:
    df[col + '_z'] = (df[col] - df[col].mean()) / df[col].std(ddof=0)

# Logistic regression with both predictors
model = smf.logit(f"{outcome} ~ rel_size_z + rel_loc_z", data=df).fit(disp=0)

# Also check each predictor separately
model_size = smf.logit(f"{outcome} ~ rel_size_z", data=df).fit(disp=0)
model_loc = smf.logit(f"{outcome} ~ rel_loc_z", data=df).fit(disp=0)

# Odds ratios and CI
params = model.params
conf = model.conf_int()
odds = np.exp(params)
conf_odds = np.exp(conf)

# Simple nonparametric checks: compare rel_size and rel_loc by win/loss
wins = df[df[outcome] == 1]
losses = df[df[outcome] == 0]

# Mann-Whitney U tests (nonparametric)
size_u = stats.mannwhitneyu(wins['rel_size'], losses['rel_size'], alternative='two-sided')
loc_u = stats.mannwhitneyu(wins['rel_loc'], losses['rel_loc'], alternative='two-sided')

# Write a small JSON with key results to parse later
results = {
    'n': len(df),
    'win_rate': df[outcome].mean(),
    'rel_size_mean_win': wins['rel_size'].mean(),
    'rel_size_mean_loss': losses['rel_size'].mean(),
    'rel_loc_mean_win': wins['rel_loc'].mean(),
    'rel_loc_mean_loss': losses['rel_loc'].mean(),
    'logit_both_params': params.to_dict(),
    'logit_both_pvalues': model.pvalues.to_dict(),
    'logit_both_odds': odds.to_dict(),
    'logit_both_odds_ci_low': conf_odds[0].to_dict(),
    'logit_both_odds_ci_high': conf_odds[1].to_dict(),
    'logit_size_pvalues': model_size.pvalues.to_dict(),
    'logit_loc_pvalues': model_loc.pvalues.to_dict(),
    'mannwhitney_rel_size_p': size_u.pvalue,
    'mannwhitney_rel_loc_p': loc_u.pvalue,
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
