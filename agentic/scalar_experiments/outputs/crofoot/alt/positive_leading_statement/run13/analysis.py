import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load data
df = pd.read_csv('crofoot.csv')

# Create predictors: relative group size and relative location
df['size_adv'] = df['n_focal'] - df['n_other']
df['loc_adv'] = df['dist_other'] - df['dist_focal']  # positive means focal closer to its center

# Standardize predictors for comparability
for col in ['size_adv', 'loc_adv']:
    df[col + '_z'] = (df[col] - df[col].mean()) / df[col].std(ddof=0)

# Logistic regression
X = df[['size_adv_z', 'loc_adv_z']]
X = sm.add_constant(X)
y = df['win']
model = sm.Logit(y, X).fit(disp=False)

# Extract results
summary = model.summary2().tables[1]

# Odds ratios and CI
params = model.params
conf = model.conf_int()
or_vals = np.exp(params)
or_ci = np.exp(conf)

# Pseudo R2
pr2 = model.prsquared

# Also fit single-predictor models for robustness
results = {}
for predictor in ['size_adv_z', 'loc_adv_z']:
    X1 = sm.add_constant(df[[predictor]])
    m = sm.Logit(y, X1).fit(disp=False)
    results[predictor] = {
        'coef': m.params[predictor],
        'p': m.pvalues[predictor],
        'or': float(np.exp(m.params[predictor])),
        'ci_low': float(np.exp(m.conf_int().loc[predictor, 0])),
        'ci_high': float(np.exp(m.conf_int().loc[predictor, 1])),
        'prsquared': m.prsquared,
    }

# Save key outputs to a JSON-like print
print('n', len(df))
print('model_params')
print(summary)
print('odds_ratios')
print(or_vals)
print('or_ci')
print(or_ci)
print('prsquared', pr2)
print('single_predictor', results)
