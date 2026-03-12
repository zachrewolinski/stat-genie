import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data
csv_path = "panda_nuts.csv"
df = pd.read_csv(csv_path)

# Basic cleaning: ensure relevant columns
# Standardize categorical values for sex and help
# Note: dataset uses 'f'/'m' and 'y'/'N' per info.json

df['sex'] = df['sex'].astype(str).str.strip().str.lower()
# help: y / n (but sample shows 'y' and 'N'); normalize to 'y'/'n'
df['help'] = df['help'].astype(str).str.strip().str.lower()

# Ensure seconds positive
# Filter rows with nonpositive seconds or missing values in relevant columns
relevant = ['age', 'sex', 'help', 'nuts_opened', 'seconds']

df = df[relevant].copy()

# Drop rows with missing values
before_rows = len(df)
df = df.dropna()

# Filter nonpositive seconds
nonpos_seconds = (df['seconds'] <= 0)
df = df.loc[~nonpos_seconds].copy()

# Create rate (efficiency)
df['rate'] = df['nuts_opened'] / df['seconds']

# Replace infinite/negative rates if any
# (shouldn't happen since seconds > 0 and nuts_opened >=0, but guard)
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# Encode categories
# Use 'sex' and 'help' as categorical in formulas

# Descriptive stats
summary = {
    'rows_initial': before_rows,
    'rows_used': len(df),
    'rate_mean': df['rate'].mean(),
    'rate_std': df['rate'].std(),
    'rate_min': df['rate'].min(),
    'rate_max': df['rate'].max(),
    'sex_counts': df['sex'].value_counts().to_dict(),
    'help_counts': df['help'].value_counts().to_dict(),
}

# OLS on rate
ols_model = smf.ols('rate ~ age + C(sex) + C(help)', data=df).fit()

# GLM Poisson on counts with offset log(seconds) to model rate
# Add small epsilon to seconds just in case
seconds = df['seconds'].astype(float)
offset = np.log(seconds)

poisson_model = smf.glm('nuts_opened ~ age + C(sex) + C(help)', data=df,
                        family=sm.families.Poisson(), offset=offset).fit()

# Overdispersion check: Pearson chi2 / df_resid
pearson_chi2 = sum(poisson_model.resid_pearson**2)
overdispersion_ratio = pearson_chi2 / poisson_model.df_resid

# Robust (HC0) standard errors for sensitivity
poisson_robust = smf.glm('nuts_opened ~ age + C(sex) + C(help)', data=df,
                         family=sm.families.Poisson(), offset=offset).fit(cov_type='HC0')

# Simple group comparisons: help effect via t-test on rate (for context)
# For help in {y,n}
help_groups = df.groupby('help')['rate']

ttest_help = None
if set(help_groups.groups.keys()) >= {'y', 'n'} or set(help_groups.groups.keys()) >= {'y','n'}:
    # map any non-'y' to 'n'
    df_help = df.copy()
    df_help['help_bin'] = np.where(df_help['help'] == 'y', 'y', 'n')
    y_rates = df_help.loc[df_help['help_bin'] == 'y', 'rate']
    n_rates = df_help.loc[df_help['help_bin'] == 'n', 'rate']
    if len(y_rates) > 1 and len(n_rates) > 1:
        ttest_help = stats.ttest_ind(y_rates, n_rates, equal_var=False, nan_policy='omit')

# Build output table of key model results

def extract_params(result):
    params = result.params
    conf = result.conf_int()
    pvalues = result.pvalues
    rows = []
    for name in params.index:
        rows.append({
            'term': name,
            'coef': float(params[name]),
            'pvalue': float(pvalues[name]),
            'conf_low': float(conf.loc[name, 0]),
            'conf_high': float(conf.loc[name, 1]),
        })
    return rows

output = {
    'summary': summary,
    'ols': {
        'r2': float(ols_model.rsquared),
        'adj_r2': float(ols_model.rsquared_adj),
        'params': extract_params(ols_model),
    },
    'poisson': {
        'aic': float(poisson_model.aic),
        'deviance': float(poisson_model.deviance),
        'df_resid': float(poisson_model.df_resid),
        'overdispersion_ratio': float(overdispersion_ratio),
        'params': extract_params(poisson_model),
    },
    'poisson_robust': {
        'params': extract_params(poisson_robust),
    },
    'ttest_help': {
        'statistic': float(ttest_help.statistic) if ttest_help is not None else None,
        'pvalue': float(ttest_help.pvalue) if ttest_help is not None else None,
        'n_help_y': int((df['help'] == 'y').sum()),
        'n_help_n': int((df['help'] != 'y').sum()),
    }
}

import json
print(json.dumps(output, indent=2))
