import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

DATA_PATH = Path('crofoot.csv')

# Load
_df = pd.read_csv(DATA_PATH)

# Define variables based on metadata
outcome = _df['feature4']  # 1 if focal won
rel_group_size = _df['feature7'] - _df['feature8']  # focal size minus other size
rel_location = _df['feature5'] - _df['feature6']  # focal distance from its center minus other distance

# Standardize predictors for comparability
X = pd.DataFrame({
    'rel_group_size': rel_group_size,
    'rel_location': rel_location,
})
X_std = (X - X.mean()) / X.std(ddof=0)
X_std = sm.add_constant(X_std)

# Logistic regression
model = sm.Logit(outcome, X_std)
result = model.fit(disp=False)

# Also compute simple bivariate models for robustness
bivar_results = {}
for col in ['rel_group_size', 'rel_location']:
    Xi = sm.add_constant(((X[[col]] - X[[col]].mean()) / X[[col]].std(ddof=0)))
    res = sm.Logit(outcome, Xi).fit(disp=False)
    bivar_results[col] = {
        'coef': float(res.params[col]),
        'pvalue': float(res.pvalues[col]),
        'odds_ratio': float(np.exp(res.params[col])),
        'ci_low': float(np.exp(res.conf_int().loc[col, 0])),
        'ci_high': float(np.exp(res.conf_int().loc[col, 1])),
    }

# Compute odds ratios and p-values for multivariate model
summary = {
    'coef': result.params.to_dict(),
    'pvalues': result.pvalues.to_dict(),
}

or_mult = np.exp(result.params)
ci_mult = np.exp(result.conf_int())

mult_results = {}
for col in ['rel_group_size', 'rel_location']:
    mult_results[col] = {
        'coef': float(result.params[col]),
        'pvalue': float(result.pvalues[col]),
        'odds_ratio': float(or_mult[col]),
        'ci_low': float(ci_mult.loc[col, 0]),
        'ci_high': float(ci_mult.loc[col, 1]),
    }

# Predicted probability change for +1 SD in each predictor (holding other at mean)
baseline = result.predict([1, 0, 0])[0]
pp_effects = {}
for col in ['rel_group_size', 'rel_location']:
    if col == 'rel_group_size':
        pred = result.predict([1, 1, 0])[0]
    else:
        pred = result.predict([1, 0, 1])[0]
    pp_effects[col] = float(pred - baseline)

analysis = {
    'n': int(len(_df)),
    'baseline_prob': float(baseline),
    'multivariate': mult_results,
    'bivariate': bivar_results,
    'pp_effects_1sd': pp_effects,
}

Path('analysis_results.json').write_text(json.dumps(analysis, indent=2))
