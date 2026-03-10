import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data
DF_PATH = 'crofoot.csv'

df = pd.read_csv(DF_PATH)

# Keep relevant columns and drop missing
cols = ['win', 'dist_focal', 'dist_other', 'n_focal', 'n_other']
sub = df[cols].copy()
sub = sub.dropna()

# Derived variables
sub['rel_size'] = sub['n_focal'] - sub['n_other']
# Positive means contest is closer to focal group's home range center
sub['loc_adv'] = sub['dist_other'] - sub['dist_focal']

# Standardize for comparable coefficients
for col in ['rel_size', 'loc_adv']:
    sub[col + '_z'] = (sub[col] - sub[col].mean()) / sub[col].std(ddof=0)

# Logistic regression with both predictors
X = sm.add_constant(sub[['rel_size_z', 'loc_adv_z']])
model = sm.Logit(sub['win'], X).fit(disp=False)

# Single-predictor models
X_size = sm.add_constant(sub[['rel_size_z']])
model_size = sm.Logit(sub['win'], X_size).fit(disp=False)

X_loc = sm.add_constant(sub[['loc_adv_z']])
model_loc = sm.Logit(sub['win'], X_loc).fit(disp=False)

# Effect sizes: odds ratios per 1 SD
or_full = np.exp(model.params)
or_size = np.exp(model_size.params)
or_loc = np.exp(model_loc.params)

# Predicted probabilities for +/-1 SD shifts
mean_row = pd.DataFrame({'const': [1.0], 'rel_size_z': [0.0], 'loc_adv_z': [0.0]})

# baseline
p_base = float(model.predict(mean_row)[0])
# +1 SD rel size
p_size_up = float(model.predict(pd.DataFrame({'const':[1.0],'rel_size_z':[1.0],'loc_adv_z':[0.0]}))[0])
# -1 SD rel size
p_size_down = float(model.predict(pd.DataFrame({'const':[1.0],'rel_size_z':[-1.0],'loc_adv_z':[0.0]}))[0])
# +1 SD location advantage
p_loc_up = float(model.predict(pd.DataFrame({'const':[1.0],'rel_size_z':[0.0],'loc_adv_z':[1.0]}))[0])
# -1 SD location advantage
p_loc_down = float(model.predict(pd.DataFrame({'const':[1.0],'rel_size_z':[0.0],'loc_adv_z':[-1.0]}))[0])

results = {
    'n_rows': int(len(sub)),
    'rel_size_mean': float(sub['rel_size'].mean()),
    'loc_adv_mean': float(sub['loc_adv'].mean()),
    'full_model': {
        'params': model.params.to_dict(),
        'pvalues': model.pvalues.to_dict(),
        'odds_ratios': or_full.to_dict(),
        'aic': float(model.aic)
    },
    'size_only': {
        'params': model_size.params.to_dict(),
        'pvalues': model_size.pvalues.to_dict(),
        'odds_ratios': or_size.to_dict(),
        'aic': float(model_size.aic)
    },
    'loc_only': {
        'params': model_loc.params.to_dict(),
        'pvalues': model_loc.pvalues.to_dict(),
        'odds_ratios': or_loc.to_dict(),
        'aic': float(model_loc.aic)
    },
    'predicted_probabilities': {
        'baseline': p_base,
        'rel_size_plus_1sd': p_size_up,
        'rel_size_minus_1sd': p_size_down,
        'loc_adv_plus_1sd': p_loc_up,
        'loc_adv_minus_1sd': p_loc_down
    }
}

print(json.dumps(results, indent=2))
