import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


df = pd.read_csv('panda_nuts.csv')

# Clean categorical variables
if 'help' in df.columns:
    df['help'] = df['help'].astype(str).str.strip().str.lower().replace({'n': 'n', 'no': 'n', 'y': 'y', 'yes': 'y'})
    # Normalize any unexpected values
    df.loc[~df['help'].isin(['y', 'n']), 'help'] = np.nan

if 'sex' in df.columns:
    df['sex'] = df['sex'].astype(str).str.strip().str.lower()
    df.loc[~df['sex'].isin(['m', 'f']), 'sex'] = np.nan

# Drop rows with missing key variables
analysis_df = df[['chimpanzee', 'age', 'sex', 'help', 'nuts_opened', 'seconds']].dropna().copy()

# Efficiency as rate
analysis_df['efficiency'] = analysis_df['nuts_opened'] / analysis_df['seconds']
analysis_df['log_seconds'] = np.log(analysis_df['seconds'])

# Poisson rate model with offset for time
model = smf.glm(
    'nuts_opened ~ age + sex + help',
    data=analysis_df,
    family=sm.families.Poisson(),
    offset=analysis_df['log_seconds'],
)

# Cluster-robust SE by chimpanzee (repeated measures)
result = model.fit(cov_type='cluster', cov_kwds={'groups': analysis_df['chimpanzee']})

# Also fit a linear model on efficiency for a sensitivity check
ols_model = smf.ols('efficiency ~ age + sex + help', data=analysis_df)
ols_result = ols_model.fit(cov_type='HC3')

output = {
    'n_rows': int(analysis_df.shape[0]),
    'poisson_params': result.params.to_dict(),
    'poisson_pvalues': result.pvalues.to_dict(),
    'poisson_rate_ratios': np.exp(result.params).to_dict(),
    'poisson_ci': result.conf_int().rename(columns={0: 'ci_lower', 1: 'ci_upper'}).to_dict(orient='index'),
    'ols_params': ols_result.params.to_dict(),
    'ols_pvalues': ols_result.pvalues.to_dict(),
    'ols_ci': ols_result.conf_int().rename(columns={0: 'ci_lower', 1: 'ci_upper'}).to_dict(orient='index'),
}

with open('analysis_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(json.dumps(output, indent=2))
