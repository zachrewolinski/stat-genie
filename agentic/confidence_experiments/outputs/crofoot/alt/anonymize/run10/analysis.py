import json
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv('crofoot.csv')

# Define variables
outcome = _df['feature4']  # 1 if focal won
size_diff = _df['feature7'] - _df['feature8']  # focal size - other size
loc_adv = _df['feature6'] - _df['feature5']    # other distance - focal distance (positive => focal closer to its home center)

# Standardize predictors for comparability
X = pd.DataFrame({
    'size_diff': size_diff,
    'loc_adv': loc_adv,
})
X_std = (X - X.mean()) / X.std(ddof=0)
X_std = sm.add_constant(X_std)

# Logistic regression
model = sm.Logit(outcome, X_std).fit(disp=False)

# Also fit univariate models to check each predictor alone
X_size = sm.add_constant(((size_diff - size_diff.mean()) / size_diff.std(ddof=0)))
model_size = sm.Logit(outcome, X_size).fit(disp=False)

X_loc = sm.add_constant(((loc_adv - loc_adv.mean()) / loc_adv.std(ddof=0)))
model_loc = sm.Logit(outcome, X_loc).fit(disp=False)

# Extract results
res = {
    'n': int(len(_df)),
    'win_rate': float(outcome.mean()),
    'coef': model.params.to_dict(),
    'pvalues': model.pvalues.to_dict(),
    'odds_ratio': np.exp(model.params).to_dict(),
    'pseudo_r2': float(model.prsquared),
    'size_only': {
        'coef': model_size.params.to_dict(),
        'pvalues': model_size.pvalues.to_dict(),
        'odds_ratio': np.exp(model_size.params).to_dict(),
        'pseudo_r2': float(model_size.prsquared),
    },
    'loc_only': {
        'coef': model_loc.params.to_dict(),
        'pvalues': model_loc.pvalues.to_dict(),
        'odds_ratio': np.exp(model_loc.params).to_dict(),
        'pseudo_r2': float(model_loc.prsquared),
    },
}

# Save intermediate results for inspection
with open('analysis_results.json', 'w') as f:
    json.dump(res, f, indent=2)

print(json.dumps(res, indent=2))
