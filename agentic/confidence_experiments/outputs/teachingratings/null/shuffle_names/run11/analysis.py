import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf


df = pd.read_csv('teachingratings.csv')

# Basic info
n = len(df)

# Outcome and predictor
y = df['allstudents']
x = df['beauty']

# Pearson correlation
corr = stats.pearsonr(x, y)

# Simple OLS
simple_model = smf.ols('allstudents ~ beauty', data=df).fit()

# Build controls: exclude division (likely identifier), and exclude outcome & predictor
exclude_cols = {'allstudents', 'beauty', 'division'}
control_cols = [c for c in df.columns if c not in exclude_cols]

# Create formula with categorical terms for object dtype
terms = []
for c in control_cols:
    if df[c].dtype == 'object':
        terms.append(f'C({c})')
    else:
        terms.append(c)

controls_formula = ' + '.join(terms)
full_formula = f'allstudents ~ beauty + {controls_formula}' if controls_formula else 'allstudents ~ beauty'

full_model = smf.ols(full_formula, data=df).fit()

# Effect size: 1 SD of beauty
beauty_sd = df['beauty'].std()
coef = simple_model.params['beauty']
coef_full = full_model.params['beauty']

# Predicted change for 1 SD in beauty
sd_effect_simple = coef * beauty_sd
sd_effect_full = coef_full * beauty_sd

# IQR effect
q1, q3 = df['beauty'].quantile([0.25, 0.75])
iqr = q3 - q1

iqr_effect_simple = coef * iqr
iqr_effect_full = coef_full * iqr

# Summaries
summary = {
    'n': n,
    'beauty_sd': beauty_sd,
    'corr_r': corr.statistic,
    'corr_p': corr.pvalue,
    'simple_coef': coef,
    'simple_p': simple_model.pvalues['beauty'],
    'simple_ci': simple_model.conf_int().loc['beauty'].tolist(),
    'full_coef': coef_full,
    'full_p': full_model.pvalues['beauty'],
    'full_ci': full_model.conf_int().loc['beauty'].tolist(),
    'sd_effect_simple': sd_effect_simple,
    'sd_effect_full': sd_effect_full,
    'iqr_effect_simple': iqr_effect_simple,
    'iqr_effect_full': iqr_effect_full,
    'full_formula': full_formula,
}

print(summary)
