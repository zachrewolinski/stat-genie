import json
import pandas as pd
import numpy as np
import statsmodels.api as sm

DATA_PATH = 'crofoot.csv'

df = pd.read_csv(DATA_PATH)

# Define variables
# Outcome: focal win (1) vs other win (0)
# Relative group size: focal size - other size
# Contest location advantage: other distance - focal distance (positive means closer to focal home range center)

df['size_diff'] = df['feature7'] - df['feature8']
df['loc_adv'] = df['feature6'] - df['feature5']

# Standardize predictors for comparability
for col in ['size_diff', 'loc_adv']:
    df[col + '_z'] = (df[col] - df[col].mean()) / df[col].std(ddof=0)

# Logistic regression
X = df[['size_diff_z', 'loc_adv_z']]
X = sm.add_constant(X)

y = df['feature4']

model = sm.Logit(y, X)
res = model.fit(disp=False)

# Odds ratios and CIs
params = res.params
conf = res.conf_int()
conf.columns = ['2.5%', '97.5%']

odds_ratios = np.exp(params)
conf_or = np.exp(conf)

# Compute pseudo R^2 (McFadden)
ll_null = sm.Logit(y, sm.add_constant(pd.DataFrame({'intercept': np.ones(len(y))}))).fit(disp=False).llf
ll_model = res.llf
pseudo_r2 = 1 - (ll_model / ll_null)

# Simple bivariate checks
# Point-biserial correlation between outcome and predictors
corrs = {}
for col in ['size_diff', 'loc_adv']:
    corrs[col] = np.corrcoef(df[col], y)[0, 1]

output = {
    'n': int(len(df)),
    'model_summary': {
        'coef': params.to_dict(),
        'pvalues': res.pvalues.to_dict(),
        'odds_ratio': odds_ratios.to_dict(),
        'odds_ratio_ci95': conf_or.to_dict(),
        'pseudo_r2_mcfadden': float(pseudo_r2),
    },
    'corrs': corrs,
}

with open('analysis_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(json.dumps(output, indent=2))
