import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('crofoot.csv')

# Define variables
# feature4: 1 if focal won, 0 if other won
# Relative group size: focal group size minus other group size
# Contest location (relative): other distance - focal distance (positive means contest closer to focal home-range center)

df = df.copy()
df['rel_group_size'] = df['feature7'] - df['feature8']
df['rel_location'] = df['feature6'] - df['feature5']

# Basic descriptive

n = len(df)

# Logistic regression models
model_both = smf.logit('feature4 ~ rel_group_size + rel_location', data=df).fit(disp=False)
model_size = smf.logit('feature4 ~ rel_group_size', data=df).fit(disp=False)
model_loc = smf.logit('feature4 ~ rel_location', data=df).fit(disp=False)

# Odds ratios and p-values

def summarize_model(model):
    params = model.params
    conf = model.conf_int()
    pvals = model.pvalues
    or_vals = np.exp(params)
    or_ci = np.exp(conf)
    return {
        'params': params.to_dict(),
        'pvalues': pvals.to_dict(),
        'odds_ratios': or_vals.to_dict(),
        'or_ci_lower': or_ci[0].to_dict(),
        'or_ci_upper': or_ci[1].to_dict(),
        'n': int(model.nobs),
        'llf': float(model.llf),
        'aic': float(model.aic),
        'bic': float(model.bic)
    }

summary = {
    'n': n,
    'rel_group_size_desc': df['rel_group_size'].describe().to_dict(),
    'rel_location_desc': df['rel_location'].describe().to_dict(),
    'model_both': summarize_model(model_both),
    'model_size': summarize_model(model_size),
    'model_loc': summarize_model(model_loc),
}

# Predicted probability change across IQR for both predictors (holding other at median)

p25_size, p75_size = np.percentile(df['rel_group_size'], [25, 75])
p25_loc, p75_loc = np.percentile(df['rel_location'], [25, 75])

median_size = df['rel_group_size'].median()
median_loc = df['rel_location'].median()

# helper

def pred_prob(size, loc):
    X = pd.DataFrame({'rel_group_size':[size], 'rel_location':[loc]})
    return float(model_both.predict(X)[0])

pp_size_low = pred_prob(p25_size, median_loc)
pp_size_high = pred_prob(p75_size, median_loc)
pp_loc_low = pred_prob(median_size, p25_loc)
pp_loc_high = pred_prob(median_size, p75_loc)

summary['pred_prob_iqr'] = {
    'rel_group_size_p25': float(p25_size),
    'rel_group_size_p75': float(p75_size),
    'rel_location_p25': float(p25_loc),
    'rel_location_p75': float(p75_loc),
    'prob_size_p25': pp_size_low,
    'prob_size_p75': pp_size_high,
    'prob_loc_p25': pp_loc_low,
    'prob_loc_p75': pp_loc_high,
}

# Print summary as json for inspection
print(json.dumps(summary, indent=2))
