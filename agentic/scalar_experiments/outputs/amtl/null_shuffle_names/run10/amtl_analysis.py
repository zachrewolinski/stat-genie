import pandas as pd
import numpy as np
import statsmodels.api as sm
from patsy import dmatrix, build_design_matrices
from scipy.stats import norm

# Load data
_df = pd.read_csv('amtl.csv')

# Map columns based on info.json metadata and observed distributions
_df['num_missing'] = _df['genus']
_df['num_sockets'] = _df['age']
_df['age_at_death'] = _df['pop']
_df['age_uncertainty'] = _df['num_amtl']
_df['prob_male_est'] = _df['stdev_age']
_df['tooth_class_cat'] = _df['sockets']
_df['genus_group'] = _df['tooth_class']

# Sanity checks
_df = _df[_df['num_sockets'] > 0].copy()
_df['num_missing'] = _df['num_missing'].clip(lower=0, upper=_df['num_sockets'])

# Binomial endog as successes/failures
endog = np.column_stack([_df['num_missing'], _df['num_sockets'] - _df['num_missing']])

# Design matrix (no intercept; genus coefficients are absolute levels)
formula = '0 + C(genus_group) + age_at_death + prob_male_est + C(tooth_class_cat)'
exog = dmatrix(formula, data=_df, return_type='dataframe')

design_info = exog.design_info

model = sm.GLM(endog, exog, family=sm.families.Binomial())
result = model.fit()

params = result.params
cov = result.cov_params()

non_human = ['Papio', 'Pan', 'Pongo']

present = [g for g in non_human if f'C(genus_group)[{g}]' in params.index]

h_key = 'C(genus_group)[Homo sapiens]'

non_human_coefs = np.array([params[f'C(genus_group)[{g}]'] for g in present])
mean_non_human = non_human_coefs.mean() if len(non_human_coefs) else 0.0

h_coef = params.get(h_key, 0.0)

log_odds_diff = h_coef - mean_non_human

coef_names = list(params.index)
contrast = np.zeros(len(coef_names))
if h_key in coef_names:
    contrast[coef_names.index(h_key)] = 1.0
if len(present) > 0:
    for g in present:
        contrast[coef_names.index(f'C(genus_group)[{g}]')] -= 1.0 / len(present)

se_diff = np.sqrt(np.dot(contrast, np.dot(cov, contrast)))

z = log_odds_diff / se_diff if se_diff > 0 else np.nan
p_value = 2 * (1 - norm.cdf(abs(z))) if np.isfinite(z) else np.nan

odds_ratio = np.exp(log_odds_diff)

# Predicted probability difference across dataset

def predict_prob(df_local, genus_label):
    df_local = df_local.copy()
    df_local['genus_group'] = genus_label
    ex = build_design_matrices([design_info], df_local, return_type='dataframe')[0]
    lin = np.dot(ex, params.values)
    return 1 / (1 + np.exp(-lin))

prob_h = predict_prob(_df, 'Homo sapiens')

if present:
    probs_non = []
    for g in present:
        probs_non.append(predict_prob(_df, g))
    prob_non = np.mean(probs_non, axis=0)
else:
    prob_non = predict_prob(_df, 'Papio')

avg_diff = float(np.mean(prob_h - prob_non))

print('log_odds_diff', log_odds_diff)
print('odds_ratio', odds_ratio)
print('z', z)
print('p_value', p_value)
print('avg_prob_diff', avg_diff)

