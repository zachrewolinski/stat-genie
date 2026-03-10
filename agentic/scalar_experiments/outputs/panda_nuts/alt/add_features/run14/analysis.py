import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'panda_nuts.csv'
df = pd.read_csv(path)

# Compute efficiency: nuts opened per second
df['efficiency'] = df['nuts_opened'] / df['seconds']

# Keep relevant columns and drop missing
cols = ['chimpanzee', 'age', 'sex', 'help', 'nuts_opened', 'seconds', 'efficiency']
subset = df[cols].copy()

# Normalize help values (strip and lower)
subset['help'] = subset['help'].astype(str).str.strip()

# Drop rows with missing or invalid values
subset = subset.replace({'': np.nan, 'nan': np.nan, 'NaN': np.nan})
subset = subset.dropna(subset=['age', 'sex', 'help', 'nuts_opened', 'seconds', 'efficiency', 'chimpanzee'])

# Ensure correct dtypes
subset['sex'] = subset['sex'].astype('category')
subset['help'] = subset['help'].astype('category')

# OLS model on efficiency with cluster-robust SE by chimpanzee
ols_model = smf.ols('efficiency ~ age + C(sex) + C(help)', data=subset)
ols_res = ols_model.fit(cov_type='cluster', cov_kwds={'groups': subset['chimpanzee']})

# Poisson GLM on nuts_opened with log(seconds) offset (rate model)
subset['log_seconds'] = np.log(subset['seconds'])
poisson_model = smf.glm(
    'nuts_opened ~ age + C(sex) + C(help)',
    data=subset,
    family=sm.families.Poisson(),
    offset=subset['log_seconds'],
)
poisson_res = poisson_model.fit(cov_type='cluster', cov_kwds={'groups': subset['chimpanzee']})

# Summaries for reporting
summary = {
    'n_obs': int(subset.shape[0]),
    'n_chimp': int(subset['chimpanzee'].nunique()),
    'efficiency_mean': float(subset['efficiency'].mean()),
    'efficiency_std': float(subset['efficiency'].std()),
    'ols_params': ols_res.params.to_dict(),
    'ols_pvalues': ols_res.pvalues.to_dict(),
    'ols_r2': float(ols_res.rsquared),
    'poisson_params': poisson_res.params.to_dict(),
    'poisson_pvalues': poisson_res.pvalues.to_dict(),
}

with open('analysis_results.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
