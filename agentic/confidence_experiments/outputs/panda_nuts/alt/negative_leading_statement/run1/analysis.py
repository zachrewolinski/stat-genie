import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
csv_path = 'panda_nuts.csv'
df = pd.read_csv(csv_path)

# Basic cleaning
# Ensure categorical columns are treated as such
for col in ['sex', 'help', 'hammer']:
    if col in df.columns:
        df[col] = df[col].astype('category')

# Efficiency: nuts opened per second
# Avoid division by zero just in case (not expected)
df['efficiency'] = df['nuts_opened'] / df['seconds'].replace({0: np.nan})

# Drop rows with missing values in key variables
key_cols = ['nuts_opened', 'seconds', 'age', 'sex', 'help']
clean = df.dropna(subset=key_cols).copy()

# Poisson GLM with offset log(seconds) to model rate
clean['log_seconds'] = np.log(clean['seconds'])

poisson_model = smf.glm(
    formula='nuts_opened ~ age + C(sex) + C(help)',
    data=clean,
    family=sm.families.Poisson(),
    offset=clean['log_seconds']
).fit(cov_type='HC3')

# Overdispersion check
pearson_chi2 = poisson_model.pearson_chi2
od_ratio = pearson_chi2 / poisson_model.df_resid if poisson_model.df_resid else np.nan

# Negative Binomial GLM (robust) as sensitivity if overdispersion
nb_model = smf.glm(
    formula='nuts_opened ~ age + C(sex) + C(help)',
    data=clean,
    family=sm.families.NegativeBinomial(alpha=1.0),
    offset=clean['log_seconds']
).fit(cov_type='HC3')

# OLS on efficiency (robust) for interpretability
ols_model = smf.ols(
    formula='efficiency ~ age + C(sex) + C(help)',
    data=clean
).fit(cov_type='HC3')

# Summaries
summary = {
    'n_rows': int(clean.shape[0]),
    'efficiency_mean': float(clean['efficiency'].mean()),
    'efficiency_sd': float(clean['efficiency'].std()),
    'poisson_params': poisson_model.params.to_dict(),
    'poisson_pvalues': poisson_model.pvalues.to_dict(),
    'poisson_conf_int': {k: list(v) for k, v in poisson_model.conf_int().iterrows()},
    'poisson_overdispersion_ratio': float(od_ratio),
    'nb_params': nb_model.params.to_dict(),
    'nb_pvalues': nb_model.pvalues.to_dict(),
    'nb_conf_int': {k: list(v) for k, v in nb_model.conf_int().iterrows()},
    'ols_params': ols_model.params.to_dict(),
    'ols_pvalues': ols_model.pvalues.to_dict(),
    'ols_conf_int': {k: list(v) for k, v in ols_model.conf_int().iterrows()},
}

print(json.dumps(summary, indent=2))
