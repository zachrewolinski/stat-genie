import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('amtl.csv')

# Create indicator for human

df['is_human'] = (df['tooth_class'] == 'Homo sapiens').astype(int)

# Model 1: OLS on AMTL count proxy (genus)
formula1 = 'genus ~ is_human + pop + stdev_age + C(sockets) + age'
model1 = smf.ols(formula1, data=df).fit(cov_type='HC3')

# Model 2: WLS on rate (genus/age) with weights=age
# Guard against zero age (none in data)

rate = df['genus'] / df['age']
df['rate'] = rate
model2 = smf.wls('rate ~ is_human + pop + stdev_age + C(sockets)', data=df, weights=df['age']).fit(cov_type='HC3')

# Model 3: OLS with genus categories to compare Homo vs non-human (baseline set to Papio for stability)
# This is mostly for robustness; we will extract Homo coefficient relative to baseline.

formula3 = 'genus ~ C(tooth_class, Treatment(reference="Papio")) + pop + stdev_age + C(sockets) + age'
model3 = smf.ols(formula3, data=df).fit(cov_type='HC3')

# Print summaries of key coefficients
print('Model1 (genus ~ human + covariates)')
print(model1.summary().tables[1])

print('\nModel2 (rate ~ human + covariates, WLS weights=age)')
print(model2.summary().tables[1])

print('\nModel3 (genus ~ genus categories + covariates)')
print(model3.summary().tables[1])

# Extract key stats

def coef_stats(model, term):
    return {
        'coef': model.params[term],
        'se': model.bse[term],
        'p': model.pvalues[term],
        'ci_low': model.conf_int().loc[term,0],
        'ci_high': model.conf_int().loc[term,1],
    }

print('\nKey stats:')
print('Model1 is_human:', coef_stats(model1, 'is_human'))
print('Model2 is_human:', coef_stats(model2, 'is_human'))
# For model3, coefficient for Homo sapiens relative to Papio
term_homo = 'C(tooth_class, Treatment(reference="Papio"))[T.Homo sapiens]'
print('Model3 Homo vs Papio:', coef_stats(model3, term_homo))

# Compute adjusted means for human vs non-human from model1 at mean covariates
means = df[['pop','stdev_age','age']].mean()
# Use reference sockets category (first in sorted order)
ref_socket = sorted(df['sockets'].unique())[0]

# Build prediction dataframe
pred_df = pd.DataFrame({
    'is_human': [0,1],
    'pop': means['pop'],
    'stdev_age': means['stdev_age'],
    'age': means['age'],
    'sockets': ref_socket,
})

pred = model1.get_prediction(pred_df).summary_frame()
print('\nPredicted genus at mean covariates (ref socket: {})'.format(ref_socket))
print(pred)

