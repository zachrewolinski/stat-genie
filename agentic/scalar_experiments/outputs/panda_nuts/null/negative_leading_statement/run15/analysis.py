import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
DF = pd.read_csv('panda_nuts.csv')

# Basic cleaning
# Normalize column names maybe; dataset uses given.

# Compute efficiency: nuts per second
DF['efficiency'] = DF['nuts_opened'] / DF['seconds']

# Some sanity checks
n = len(DF)

# Fit OLS with categorical sex/help
model = smf.ols('efficiency ~ age + C(sex) + C(help)', data=DF).fit(cov_type='HC3')

# Also check if using log efficiency helps (avoid log(0))
DF['efficiency_adj'] = DF['efficiency'].replace(0, np.nan)
DF['log_efficiency'] = np.log(DF['efficiency_adj'])
log_df = DF.dropna(subset=['log_efficiency']).copy()
log_model = smf.ols('log_efficiency ~ age + C(sex) + C(help)', data=log_df).fit(cov_type='HC3')

# Group means for interpretability
mean_by_sex = DF.groupby('sex')['efficiency'].mean().to_dict()
mean_by_help = DF.groupby('help')['efficiency'].mean().to_dict()

# Simple correlations
corr_age_eff = DF[['age', 'efficiency']].corr().iloc[0,1]

# Collect results
out = {
    'n_rows': n,
    'efficiency_summary': DF['efficiency'].describe().to_dict(),
    'mean_by_sex': mean_by_sex,
    'mean_by_help': mean_by_help,
    'corr_age_eff': corr_age_eff,
    'model_params': model.params.to_dict(),
    'model_pvalues': model.pvalues.to_dict(),
    'model_rsquared': model.rsquared,
    'log_model_params': log_model.params.to_dict(),
    'log_model_pvalues': log_model.pvalues.to_dict(),
    'log_model_rsquared': log_model.rsquared,
}

print(json.dumps(out, indent=2))
