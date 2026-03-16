import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
_df = pd.read_csv('panda_nuts.csv')

# Compute efficiency: nuts opened per second
_df['efficiency'] = _df['nuts_opened'] / _df['seconds']

# Basic model
model = smf.ols('efficiency ~ age + C(sex) + C(help)', data=_df).fit()

# Also check model controlling for hammer type as robustness
model_hammer = smf.ols('efficiency ~ age + C(sex) + C(help) + C(hammer)', data=_df).fit()

# Standardized coefficients (z-score continuous, effect-coded? We'll standardize outcome and age, keep dummies)
_df_std = _df.copy()
_df_std['efficiency_z'] = (_df_std['efficiency'] - _df_std['efficiency'].mean()) / _df_std['efficiency'].std(ddof=0)
_df_std['age_z'] = (_df_std['age'] - _df_std['age'].mean()) / _df_std['age'].std(ddof=0)
model_std = smf.ols('efficiency_z ~ age_z + C(sex) + C(help)', data=_df_std).fit()

# Summaries
print('N:', len(_df))
print('Efficiency summary:', _df['efficiency'].describe())
print('\nModel (no hammer):')
print(model.summary())
print('\nModel (with hammer):')
print(model_hammer.summary())
print('\nModel standardized (age standardized):')
print(model_std.summary())

# Save key stats for downstream
results = {
    'n': len(_df),
    'efficiency_mean': float(_df['efficiency'].mean()),
    'efficiency_std': float(_df['efficiency'].std(ddof=1)),
    'model_params': model.params.to_dict(),
    'model_pvalues': model.pvalues.to_dict(),
    'model_r2': float(model.rsquared),
    'model_adj_r2': float(model.rsquared_adj),
    'model_f_pvalue': float(model.f_pvalue),
    'model_hammer_params': model_hammer.params.to_dict(),
    'model_hammer_pvalues': model_hammer.pvalues.to_dict(),
    'model_hammer_r2': float(model_hammer.rsquared),
    'model_hammer_adj_r2': float(model_hammer.rsquared_adj),
    'model_hammer_f_pvalue': float(model_hammer.f_pvalue),
    'model_std_params': model_std.params.to_dict(),
    'model_std_pvalues': model_std.pvalues.to_dict(),
}

import json
with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)
