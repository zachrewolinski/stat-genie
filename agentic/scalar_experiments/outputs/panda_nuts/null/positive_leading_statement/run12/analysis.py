import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from pathlib import Path

# Load data
csv_path = Path('panda_nuts.csv')
df = pd.read_csv(csv_path)

# Basic cleaning
# Standardize help labels maybe 'y' and 'N' - ensure consistent casing
if df['help'].dtype == object:
    df['help'] = df['help'].str.strip().str.lower()

# Standardize sex labels
if df['sex'].dtype == object:
    df['sex'] = df['sex'].str.strip().str.lower()

# Efficiency (nuts per second)
df['efficiency'] = df['nuts_opened'] / df['seconds']

# Remove any rows with missing or zero seconds
analysis_df = df.dropna(subset=['nuts_opened','seconds','age','sex','help']).copy()
analysis_df = analysis_df[analysis_df['seconds'] > 0]

# OLS on efficiency
ols_model = smf.ols('efficiency ~ age + C(sex) + C(help)', data=analysis_df).fit()

# Poisson GLM on counts with offset log(seconds)
# Add small epsilon to seconds to avoid log(0)
analysis_df['log_seconds'] = np.log(analysis_df['seconds'])
poisson_model = smf.glm('nuts_opened ~ age + C(sex) + C(help)', data=analysis_df,
                        family=sm.families.Poisson(), offset=analysis_df['log_seconds']).fit()

# Overdispersion check: if variance >> mean, consider negative binomial
mean_counts = analysis_df['nuts_opened'].mean()
var_counts = analysis_df['nuts_opened'].var(ddof=1)

# Negative binomial GLM if overdispersed
nb_model = None
if var_counts > mean_counts * 1.5:
    try:
        nb_model = smf.glm('nuts_opened ~ age + C(sex) + C(help)', data=analysis_df,
                           family=sm.families.NegativeBinomial(), offset=analysis_df['log_seconds']).fit()
    except Exception:
        nb_model = None

# Collect results
results = {
    'n': len(analysis_df),
    'mean_efficiency': analysis_df['efficiency'].mean(),
    'ols_params': ols_model.params.to_dict(),
    'ols_pvalues': ols_model.pvalues.to_dict(),
    'poisson_params': poisson_model.params.to_dict(),
    'poisson_pvalues': poisson_model.pvalues.to_dict(),
    'mean_counts': float(mean_counts),
    'var_counts': float(var_counts),
}
if nb_model is not None:
    results['nb_params'] = nb_model.params.to_dict()
    results['nb_pvalues'] = nb_model.pvalues.to_dict()

print(json.dumps(results, indent=2))
