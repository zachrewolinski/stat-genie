import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
path = 'panda_nuts.csv'
df = pd.read_csv(path)

# Rename columns for clarity
col_map = {
    'feature1': 'id',
    'feature2': 'age',
    'feature3': 'sex',
    'feature4': 'hammer',
    'feature5': 'nuts_opened',
    'feature6': 'duration_sec',
    'feature7': 'help'
}
df = df.rename(columns=col_map)

# Compute efficiency: nuts opened per second
# Guard against any zero or negative durations
if (df['duration_sec'] <= 0).any():
    df = df[df['duration_sec'] > 0].copy()

df['efficiency'] = df['nuts_opened'] / df['duration_sec']

# Basic stats
summary = {
    'n_rows': len(df),
    'efficiency_mean': df['efficiency'].mean(),
    'efficiency_std': df['efficiency'].std(),
    'efficiency_min': df['efficiency'].min(),
    'efficiency_max': df['efficiency'].max(),
}

# OLS model with age + sex + help
ols = smf.ols('efficiency ~ age + C(sex) + C(help)', data=df).fit(cov_type='HC3')

# Mixed effects model with random intercept by individual
mixed = None
try:
    mixed = smf.mixedlm('efficiency ~ age + C(sex) + C(help)', data=df, groups=df['id']).fit(reml=False)
except Exception as e:
    mixed = str(e)

# Also test log efficiency to reduce skew
# Add a small constant to avoid log(0)
df['log_eff'] = np.log(df['efficiency'] + 1e-6)
ols_log = smf.ols('log_eff ~ age + C(sex) + C(help)', data=df).fit(cov_type='HC3')

results = {
    'summary': summary,
    'ols_params': ols.params.to_dict(),
    'ols_pvalues': ols.pvalues.to_dict(),
    'ols_r2': float(ols.rsquared),
    'ols_log_params': ols_log.params.to_dict(),
    'ols_log_pvalues': ols_log.pvalues.to_dict(),
    'ols_log_r2': float(ols_log.rsquared),
    'mixed_params': None,
    'mixed_pvalues': None,
    'mixed_aic': None,
    'mixed_status': None,
}

if isinstance(mixed, str):
    results['mixed_status'] = mixed
else:
    results['mixed_params'] = mixed.params.to_dict()
    try:
        results['mixed_pvalues'] = mixed.pvalues.to_dict()
    except Exception:
        results['mixed_pvalues'] = None
    results['mixed_aic'] = float(mixed.aic)
    results['mixed_status'] = 'ok'

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
