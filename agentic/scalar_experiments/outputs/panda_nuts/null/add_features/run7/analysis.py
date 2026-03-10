import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('panda_nuts.csv')

# Clean/prepare variables
# Normalize help and sex
help_map = {'y': 1, 'yes': 1, 'n': 0, 'no': 0}
sex_map = {'f': 1, 'female': 1, 'm': 0, 'male': 0}


df['help_bin'] = (
    df['help']
    .astype(str)
    .str.strip()
    .str.lower()
    .map(help_map)
)

df['sex_bin'] = (
    df['sex']
    .astype(str)
    .str.strip()
    .str.lower()
    .map(sex_map)
)

# Efficiency as nuts opened per second
# Avoid divide-by-zero (seconds should be > 0)

df = df[df['seconds'] > 0].copy()

df['efficiency'] = df['nuts_opened'] / df['seconds']

# Keep rows with complete data
analysis_cols = ['nuts_opened', 'seconds', 'age', 'sex_bin', 'help_bin', 'efficiency']
analysis_df = df.dropna(subset=analysis_cols).copy()

# OLS on efficiency with robust SEs
ols = smf.ols('efficiency ~ age + sex_bin + help_bin', data=analysis_df).fit()
ols_robust = ols.get_robustcov_results(cov_type='HC3')

# Poisson GLM on counts with log(seconds) offset
analysis_df['log_seconds'] = np.log(analysis_df['seconds'])
poisson = smf.glm(
    'nuts_opened ~ age + sex_bin + help_bin',
    data=analysis_df,
    family=sm.families.Poisson(),
    offset=analysis_df['log_seconds']
).fit()

# Check overdispersion
poisson_overdispersion = poisson.deviance / poisson.df_resid

# Negative binomial as sensitivity if overdispersion
nb = smf.glm(
    'nuts_opened ~ age + sex_bin + help_bin',
    data=analysis_df,
    family=sm.families.NegativeBinomial(alpha=1.0),
    offset=analysis_df['log_seconds']
).fit()

# Summaries of key results
results = {
    'n': int(len(analysis_df)),
    'ols_coef': dict(zip(ols_robust.model.exog_names, ols_robust.params)),
    'ols_pvalues': dict(zip(ols_robust.model.exog_names, ols_robust.pvalues)),
    'poisson_coef': dict(zip(poisson.model.exog_names, poisson.params)),
    'poisson_pvalues': dict(zip(poisson.model.exog_names, poisson.pvalues)),
    'poisson_overdispersion': float(poisson_overdispersion),
    'nb_coef': dict(zip(nb.model.exog_names, nb.params)),
    'nb_pvalues': dict(zip(nb.model.exog_names, nb.pvalues)),
}

# Save analysis results for inspection
with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Print concise output
print('N:', results['n'])
print('OLS (robust) p-values:', results['ols_pvalues'])
print('Poisson p-values:', results['poisson_pvalues'])
print('Poisson overdispersion:', results['poisson_overdispersion'])
print('NB p-values:', results['nb_pvalues'])
