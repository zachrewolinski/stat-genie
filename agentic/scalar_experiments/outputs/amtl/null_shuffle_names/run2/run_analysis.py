import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# load data
_df = pd.read_csv('amtl.csv')

# Map columns (based on metadata + inspection)
# sockets: tooth class
# prob_male: specimen ID (unused)
# genus: number of missing teeth (AMTL count)
# age: number of observable sockets (denominator)
# pop: estimated age at death
# num_amtl: age uncertainty (unused)
# stdev_age: probability male (sex estimate)
# tooth_class: genus (Homo sapiens, Pan, Pongo, Papio)
# specimen: population/region (unused)

# prepare data
_df = _df.copy()

# Rename for clarity
_df = _df.rename(columns={
    'prob_male': 'specimen_id',
    'genus': 'amtl_missing',
    'age': 'sockets_observed',
    'pop': 'age_at_death',
    'stdev_age': 'prob_male',
    'sockets': 'tooth_class',
    'tooth_class': 'genus'
})

# Filter out any rows with impossible counts
_df = _df[_df['sockets_observed'] > 0].copy()
_df['amtl_missing'] = _df['amtl_missing'].astype(float)
_df['sockets_observed'] = _df['sockets_observed'].astype(float)

# Cap missing to sockets if slight violations, but keep record
violations = (_df['amtl_missing'] > _df['sockets_observed']).sum()

# For binomial, enforce missing <= sockets by clipping small number of violations
_df['amtl_missing_clipped'] = _df['amtl_missing'].clip(upper=_df['sockets_observed'])

# create proportion
_df['amtl_rate'] = _df['amtl_missing_clipped'] / _df['sockets_observed']

# predictor: human vs non-human
_df['is_human'] = (_df['genus'] == 'Homo sapiens').astype(int)

# build model: binomial GLM with weights = sockets
formula = 'amtl_rate ~ is_human + age_at_death + prob_male + C(tooth_class)'
model = smf.glm(
    formula=formula,
    data=_df,
    family=sm.families.Binomial(),
    freq_weights=_df['sockets_observed']
).fit()

# Extract effect for is_human
coef = model.params['is_human']
se = model.bse['is_human']

# compute odds ratio and p-value
odds_ratio = np.exp(coef)
p_value = model.pvalues['is_human']

# predicted marginal difference at mean covariates
mean_row = {
    'is_human': 0,
    'age_at_death': _df['age_at_death'].mean(),
    'prob_male': _df['prob_male'].mean(),
}

# set tooth_class to most common for prediction
most_common_tooth = _df['tooth_class'].value_counts().idxmax()

# build design rows
pred_df_non = pd.DataFrame([{**mean_row, 'tooth_class': most_common_tooth}])
pred_df_human = pred_df_non.copy()
pred_df_human['is_human'] = 1

pred_non = model.predict(pred_df_non)[0]
pred_human = model.predict(pred_df_human)[0]

# print summary info
print('violations (missing > sockets):', int(violations))
print('is_human coef:', coef)
print('is_human odds_ratio:', odds_ratio)
print('p_value:', p_value)
print('predicted rate non-human:', pred_non)
print('predicted rate human:', pred_human)
print('predicted diff (human - non-human):', pred_human - pred_non)
