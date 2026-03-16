import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from pathlib import Path
import json

path = Path('teachingratings.csv')

df = pd.read_csv(path)

# Key variables
beauty = df['beauty']
ratings = df['allstudents']

# Pearson correlation
corr = beauty.corr(ratings)

# Simple OLS
model_simple = ols('allstudents ~ beauty', data=df).fit()

# Build controls (exclude likely identifiers)
# Treat categorical columns as categories
cat_cols = ['eval','tenure','prof','native','gender','credits']

# Numeric controls (exclude division if it seems like an id)
num_cols = ['age','rownames','minority','students']

# Drop rows with missing (none expected)

# Construct formula
formula = 'allstudents ~ beauty'
for col in num_cols:
    if col in df.columns:
        formula += f' + {col}'
for col in cat_cols:
    if col in df.columns:
        formula += f' + C({col})'

model_controls = ols(formula, data=df).fit()

# Robust SE (HC3) for controls model
model_controls_robust = model_controls.get_robustcov_results(cov_type='HC3')
model_simple_robust = model_simple.get_robustcov_results(cov_type='HC3')

# Effect sizes
beauty_sd = beauty.std()
coef_simple = model_simple.params['beauty']
coef_controls = model_controls.params['beauty']

# Effect for 1 SD change in beauty
sd_effect_simple = coef_simple * beauty_sd
sd_effect_controls = coef_controls * beauty_sd

# IQR effect
beauty_q1 = beauty.quantile(0.25)
beauty_q3 = beauty.quantile(0.75)
IQR = beauty_q3 - beauty_q1
IQR_effect_simple = coef_simple * IQR
IQR_effect_controls = coef_controls * IQR

results = {
    'n': int(df.shape[0]),
    'corr': corr,
    'simple': {
        'coef': coef_simple,
        'pvalue': model_simple.pvalues['beauty'],
        'stderr': model_simple.bse['beauty'],
        'r2': model_simple.rsquared,
        'coef_robust_se': model_simple_robust.bse[1],
        'pvalue_robust': model_simple_robust.pvalues[1],
    },
    'controls': {
        'formula': formula,
        'coef': coef_controls,
        'pvalue': model_controls.pvalues['beauty'],
        'stderr': model_controls.bse['beauty'],
        'r2': model_controls.rsquared,
        'coef_robust_se': model_controls_robust.bse[model_controls_robust.model.exog_names.index('beauty')],
        'pvalue_robust': model_controls_robust.pvalues[model_controls_robust.model.exog_names.index('beauty')],
    },
    'beauty_sd': beauty_sd,
    'sd_effect_simple': sd_effect_simple,
    'sd_effect_controls': sd_effect_controls,
    'IQR': IQR,
    'IQR_effect_simple': IQR_effect_simple,
    'IQR_effect_controls': IQR_effect_controls,
    'beauty_q1': beauty_q1,
    'beauty_q3': beauty_q3,
}

print(json.dumps(results, indent=2))
