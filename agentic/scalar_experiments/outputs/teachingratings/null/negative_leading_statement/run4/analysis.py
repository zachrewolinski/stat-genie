import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
csv_path = 'teachingratings.csv'
df = pd.read_csv(csv_path)

# Basic cleaning: drop rows with missing key vars
key_vars = ['beauty', 'eval']
sub = df.dropna(subset=key_vars).copy()

# Summary stats
n = len(sub)
beauty_mean = sub['beauty'].mean()
beauty_sd = sub['beauty'].std(ddof=1)
eval_mean = sub['eval'].mean()
eval_sd = sub['eval'].std(ddof=1)

# Bivariate correlation
corr = sub['beauty'].corr(sub['eval'])

# Simple OLS
model_simple = smf.ols('eval ~ beauty', data=sub).fit()

# Controlled OLS with common covariates
# Use available columns, treat categorical appropriately
formula = (
    'eval ~ beauty + age + C(gender) + C(minority) + C(credits) + '
    'C(division) + C(native) + C(tenure) + students + allstudents'
)
model_ctrl = smf.ols(formula, data=sub).fit()

# Extract key stats
beauty_coef_simple = model_simple.params.get('beauty', np.nan)
beauty_p_simple = model_simple.pvalues.get('beauty', np.nan)
beauty_coef_ctrl = model_ctrl.params.get('beauty', np.nan)
beauty_p_ctrl = model_ctrl.pvalues.get('beauty', np.nan)

# Effect of 1 SD beauty
sd_effect_simple = beauty_coef_simple * beauty_sd
sd_effect_ctrl = beauty_coef_ctrl * beauty_sd

# Prepare results for inspection
results = {
    'n': n,
    'beauty_mean': beauty_mean,
    'beauty_sd': beauty_sd,
    'eval_mean': eval_mean,
    'eval_sd': eval_sd,
    'corr_beauty_eval': corr,
    'simple': {
        'coef': beauty_coef_simple,
        'pval': beauty_p_simple,
        'r2': model_simple.rsquared,
        'sd_effect': sd_effect_simple,
    },
    'controlled': {
        'coef': beauty_coef_ctrl,
        'pval': beauty_p_ctrl,
        'r2': model_ctrl.rsquared,
        'sd_effect': sd_effect_ctrl,
    },
}

print(json.dumps(results, indent=2))
