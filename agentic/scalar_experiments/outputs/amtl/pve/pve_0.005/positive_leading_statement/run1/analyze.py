import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
path = 'amtl.csv'
df = pd.read_csv(path)

# Basic derived variables
# AMTL rate per socket (may be noisy due to privacy perturbation)
df['amtl_rate'] = df['num_amtl'] / df['sockets']

# Center age to improve interpretation
age_mean = df['age'].mean()
df['age_c'] = df['age'] - age_mean

# Fit linear model on num_amtl (primary) with controls
formula = 'num_amtl ~ C(genus) + age_c + prob_male + C(tooth_class)'
model = smf.ols(formula, data=df).fit()

# Also fit on rate as a sensitivity check
rate_formula = 'amtl_rate ~ C(genus) + age_c + prob_male + C(tooth_class)'
rate_model = smf.ols(rate_formula, data=df).fit()

# Extract genus coefficients relative to reference (alphabetical by default)
# To get Homo sapiens vs others, change reference to Homo sapiens
model_homo_ref = smf.ols('num_amtl ~ C(genus, Treatment(reference="Homo sapiens")) + age_c + prob_male + C(tooth_class)', data=df).fit()
rate_model_homo_ref = smf.ols('amtl_rate ~ C(genus, Treatment(reference="Homo sapiens")) + age_c + prob_male + C(tooth_class)', data=df).fit()

# Summaries
print('N', len(df))
print('Genus counts')
print(df['genus'].value_counts())
print('\nModel (num_amtl) reference:', model.model.data.orig_exog['C(genus)[T.Pan]'].name if 'C(genus)[T.Pan]' in model.params else 'default')
print(model.summary().tables[1])

print('\nModel (num_amtl) with Homo sapiens as reference')
print(model_homo_ref.summary().tables[1])

print('\nModel (amtl_rate) with Homo sapiens as reference')
print(rate_model_homo_ref.summary().tables[1])

# Compute predicted means by genus at average covariates for num_amtl model
# Build a small data frame with average covariates and each genus
avg_age_c = 0.0
avg_prob_male = df['prob_male'].mean()
# Use most common tooth_class for simplicity
common_tooth = df['tooth_class'].mode()[0]

pred_df = pd.DataFrame({
    'genus': df['genus'].unique(),
    'age_c': avg_age_c,
    'prob_male': avg_prob_male,
    'tooth_class': common_tooth,
})

preds = model.get_prediction(pred_df).summary_frame(alpha=0.05)

pred_df = pd.concat([pred_df, preds[['mean','mean_ci_lower','mean_ci_upper']]], axis=1)
print('\nPredicted num_amtl at average covariates and tooth_class=', common_tooth)
print(pred_df.sort_values('mean', ascending=False))

# Pairwise differences Homo vs others (num_amtl)
# Using model with Homo reference: coefficients for other genera are differences vs Homo
coef = model_homo_ref.params
pvals = model_homo_ref.pvalues
print('\nDifferences vs Homo (num_amtl):')
for term in coef.index:
    if term.startswith('C(genus, Treatment(reference="Homo sapiens"))'):
        print(term, 'coef', coef[term], 'p', pvals[term])

# Same for rate
coef_r = rate_model_homo_ref.params
pvals_r = rate_model_homo_ref.pvalues
print('\nDifferences vs Homo (amtl_rate):')
for term in coef_r.index:
    if term.startswith('C(genus, Treatment(reference="Homo sapiens"))'):
        print(term, 'coef', coef_r[term], 'p', pvals_r[term])
