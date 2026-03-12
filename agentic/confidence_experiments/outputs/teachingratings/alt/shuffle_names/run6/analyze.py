import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

path = 'teachingratings.csv'

df = pd.read_csv(path)

# Basic cleaning
subset = df[['beauty', 'allstudents']].dropna()

n = len(subset)

# Correlations
pearson_r, pearson_p = stats.pearsonr(subset['beauty'], subset['allstudents'])
spearman_r, spearman_p = stats.spearmanr(subset['beauty'], subset['allstudents'])

# Simple OLS
model_simple = smf.ols('allstudents ~ beauty', data=df).fit()

# Build a control model with available covariates
# Treat object columns as categorical in formula
cat_cols = [c for c in df.columns if df[c].dtype == 'object' and c not in ['allstudents']]
# Numeric controls (excluding outcome, beauty)
num_cols = [c for c in df.columns if df[c].dtype != 'object' and c not in ['allstudents', 'beauty']]

# Construct formula
cat_terms = [f'C({c})' for c in cat_cols]
all_terms = ['beauty'] + num_cols + cat_terms
formula = 'allstudents ~ ' + ' + '.join(all_terms)

model_controls = smf.ols(formula, data=df).fit()

# Effect size: change in rating per 1 SD increase in beauty
beauty_sd = df['beauty'].std()
coef_simple = model_simple.params['beauty']
coef_controls = model_controls.params['beauty']

# Predicted rating change per 1 SD beauty
sd_effect_simple = coef_simple * beauty_sd
sd_effect_controls = coef_controls * beauty_sd

# 95% CI for beauty coefficient
ci_simple = model_simple.conf_int().loc['beauty'].tolist()
ci_controls = model_controls.conf_int().loc['beauty'].tolist()

results = {
    'n': n,
    'pearson_r': pearson_r,
    'pearson_p': pearson_p,
    'spearman_r': spearman_r,
    'spearman_p': spearman_p,
    'simple_coef': coef_simple,
    'simple_p': model_simple.pvalues['beauty'],
    'simple_ci': ci_simple,
    'controls_coef': coef_controls,
    'controls_p': model_controls.pvalues['beauty'],
    'controls_ci': ci_controls,
    'sd_effect_simple': sd_effect_simple,
    'sd_effect_controls': sd_effect_controls,
    'r2_simple': model_simple.rsquared,
    'r2_controls': model_controls.rsquared,
    'formula_controls': formula,
}

for k, v in results.items():
    print(f'{k}: {v}')
