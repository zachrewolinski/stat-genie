import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

DATA_PATH = 'panda_nuts.csv'

df = pd.read_csv(DATA_PATH)

# Rename for clarity
col_id = 'feature1'
col_age = 'feature2'
col_sex = 'feature3'
col_hammer = 'feature4'
col_nuts = 'feature5'
col_duration = 'feature6'
col_help = 'feature7'

# Basic cleaning
# Ensure duration > 0
if (df[col_duration] <= 0).any():
    raise ValueError('Non-positive durations found')

# Exposure in seconds; efficiency as nuts per minute
rate_per_min = df[col_nuts] / df[col_duration] * 60.0

# Add columns
analysis_df = df.copy()
analysis_df['rate_per_min'] = rate_per_min
analysis_df['log_rate_per_min_plus1'] = np.log1p(rate_per_min)
analysis_df['log_duration'] = np.log(df[col_duration])

# Poisson GLM with offset for duration to model nuts opened
# Use age, sex, help as predictors
formula = f"{col_nuts} ~ {col_age} + C({col_sex}) + C({col_help})"
poisson_model = smf.glm(
    formula=formula,
    data=analysis_df,
    family=sm.families.Poisson(),
    offset=analysis_df['log_duration']
).fit()

# Overdispersion check: Pearson chi2 / df_resid
pearson_chi2 = poisson_model.pearson_chi2
od_ratio = pearson_chi2 / poisson_model.df_resid

# If overdispersion > 1.5, fit Negative Binomial
use_nb = od_ratio > 1.5
nb_model = None
if use_nb:
    nb_model = smf.glm(
        formula=formula,
        data=analysis_df,
        family=sm.families.NegativeBinomial(alpha=1.0),
        offset=analysis_df['log_duration']
    ).fit()

# OLS on log rate (robust SE) as a sensitivity analysis
ols_model = smf.ols(
    formula=f"log_rate_per_min_plus1 ~ {col_age} + C({col_sex}) + C({col_help})",
    data=analysis_df
).fit(cov_type='HC3')

# Collect results

def summarize_model(model):
    params = model.params
    conf = model.conf_int()
    pvals = model.pvalues
    return {
        'params': params.to_dict(),
        'conf_int': conf.rename(columns={0: 'low', 1: 'high'}).to_dict(orient='index'),
        'pvalues': pvals.to_dict(),
        'aic': model.aic,
        'df_resid': float(model.df_resid),
    }

results = {
    'n': int(len(df)),
    'poisson': summarize_model(poisson_model),
    'overdispersion_ratio': float(od_ratio),
    'used_negative_binomial': bool(use_nb),
    'negative_binomial': summarize_model(nb_model) if nb_model is not None else None,
    'ols_log_rate': summarize_model(ols_model),
    'rate_per_min_summary': {
        'mean': float(rate_per_min.mean()),
        'std': float(rate_per_min.std(ddof=1)),
        'min': float(rate_per_min.min()),
        'max': float(rate_per_min.max()),
    },
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('Saved analysis_results.json')
