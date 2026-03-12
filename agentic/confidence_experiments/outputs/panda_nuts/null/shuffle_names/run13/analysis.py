import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy
from statsmodels.discrete.discrete_model import NegativeBinomial

# Load data
path = 'panda_nuts.csv'
df = pd.read_csv(path)

# Map columns based on metadata/values
# nuts_opened column contains m/f -> sex
# sex column contains hammer type -> not used
# help column contains count of nuts opened
# chimpanzee column contains session duration (seconds)
# seconds column contains help indicator (y/N)

df = df.copy()

df['sex_mf'] = df['nuts_opened'].astype(str).str.strip()
df['helped'] = df['seconds'].astype(str).str.strip()

df['nuts_opened_count'] = pd.to_numeric(df['help'], errors='coerce')
df['duration_seconds'] = pd.to_numeric(df['chimpanzee'], errors='coerce')
df['age_years'] = pd.to_numeric(df['age'], errors='coerce')

# Drop rows with missing essentials
analysis_df = df.dropna(subset=['sex_mf', 'helped', 'nuts_opened_count', 'duration_seconds', 'age_years']).copy()

# Efficiency as rate
analysis_df['rate'] = analysis_df['nuts_opened_count'] / analysis_df['duration_seconds']
analysis_df['log_duration'] = np.log(analysis_df['duration_seconds'])

# Descriptive stats
rate_overall = analysis_df['rate'].describe()
rate_by_sex = analysis_df.groupby('sex_mf')['rate'].mean().to_dict()
rate_by_help = analysis_df.groupby('helped')['rate'].mean().to_dict()

# Overdispersion check for count model
mean_count = analysis_df['nuts_opened_count'].mean()
var_count = analysis_df['nuts_opened_count'].var()
overdisp_ratio = var_count / mean_count if mean_count > 0 else np.nan

# Negative Binomial with estimated alpha (NB2)
# Use patsy to build design matrix with categorical terms
formula = 'nuts_opened_count ~ age_years + C(sex_mf) + C(helped)'

y, X = patsy.dmatrices(formula, data=analysis_df, return_type='dataframe')
nb_model = NegativeBinomial(y, X, offset=analysis_df['log_duration'])
nb_result = nb_model.fit(disp=False)

# OLS on rate (robust SE) as a check
ols = smf.ols('rate ~ age_years + C(sex_mf) + C(helped)', data=analysis_df).fit(cov_type='HC3')

# Compute rate ratios for NB
nb_params = nb_result.params
nb_pvalues = nb_result.pvalues
rate_ratios = np.exp(nb_params)

summary = {
    'n': int(analysis_df.shape[0]),
    'mean_rate': float(rate_overall['mean']),
    'rate_by_sex': rate_by_sex,
    'rate_by_help': rate_by_help,
    'overdisp_ratio': float(overdisp_ratio) if np.isfinite(overdisp_ratio) else None,
    'nb_params': nb_params.to_dict(),
    'nb_pvalues': nb_pvalues.to_dict(),
    'nb_rate_ratios': rate_ratios.to_dict(),
    'ols_params': ols.params.to_dict(),
    'ols_pvalues': ols.pvalues.to_dict(),
}

print(json.dumps(summary, indent=2, sort_keys=True))
