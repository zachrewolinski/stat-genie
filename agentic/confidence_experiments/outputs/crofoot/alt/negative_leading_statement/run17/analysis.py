import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('crofoot.csv')

# Derived variables
# Relative group size (focal minus other)
df['rel_size'] = df['n_focal'] - df['n_other']

# Location advantage: positive if contest closer to focal group's center
# Using distance to home range centers
# Smaller distance means closer; so advantage = dist_other - dist_focal
# If positive, focal is closer to its own center than other group is to theirs

df['loc_adv'] = df['dist_other'] - df['dist_focal']

# Standardize predictors for comparability
for col in ['rel_size', 'loc_adv']:
    df[f'z_{col}'] = (df[col] - df[col].mean()) / df[col].std(ddof=0)

# Logistic regression with both predictors
model = smf.glm('win ~ z_rel_size + z_loc_adv', data=df, family=sm.families.Binomial()).fit()

# Individual predictor models
model_size = smf.glm('win ~ z_rel_size', data=df, family=sm.families.Binomial()).fit()
model_loc = smf.glm('win ~ z_loc_adv', data=df, family=sm.families.Binomial()).fit()

# Odds ratios and 95% CI for combined model
params = model.params
conf = model.conf_int()
odds = np.exp(params)
conf_odds = np.exp(conf)

# Summaries
print('N:', len(df))
print('win rate:', df['win'].mean())
print('\nCombined model:')
print(model.summary())
print('\nOdds ratios (combined model):')
for term in params.index:
    print(term, 'OR=', odds[term], 'CI=', tuple(conf_odds.loc[term]))

print('\nSize-only model:')
print(model_size.summary())
print('\nLocation-only model:')
print(model_loc.summary())

# Save key stats to json-ish output for parsing
import json

out = {
    'n': int(len(df)),
    'win_rate': float(df['win'].mean()),
    'combined': {
        'params': params.to_dict(),
        'pvalues': model.pvalues.to_dict(),
        'odds_ratios': odds.to_dict(),
        'conf_int': conf.to_dict(),
        'conf_int_odds': conf_odds.to_dict(),
        'llf': float(model.llf),
        'aic': float(model.aic),
    },
    'size_only': {
        'params': model_size.params.to_dict(),
        'pvalues': model_size.pvalues.to_dict(),
        'aic': float(model_size.aic),
    },
    'loc_only': {
        'params': model_loc.params.to_dict(),
        'pvalues': model_loc.pvalues.to_dict(),
        'aic': float(model_loc.aic),
    },
    'summary_stats': {
        'rel_size_mean': float(df['rel_size'].mean()),
        'rel_size_sd': float(df['rel_size'].std(ddof=0)),
        'loc_adv_mean': float(df['loc_adv'].mean()),
        'loc_adv_sd': float(df['loc_adv'].std(ddof=0)),
    }
}

with open('analysis_results.json', 'w') as f:
    json.dump(out, f, indent=2)

