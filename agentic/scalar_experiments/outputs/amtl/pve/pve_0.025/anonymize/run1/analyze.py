import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('amtl.csv')

# Rename columns
cols = {
    'feature1':'tooth_class',
    'feature2':'specimen_id',
    'feature3':'missing',
    'feature4':'sockets',
    'feature5':'age',
    'feature6':'age_uncert',
    'feature7':'sex',
    'feature8':'genus',
    'feature9':'region'
}

df = df.rename(columns=cols)

# Derived variables

df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

df['rate'] = df['missing'] / df['sockets']

# Summary stats
summary = {
    'n': len(df),
    'missing_min': df['missing'].min(),
    'missing_max': df['missing'].max(),
    'sockets_min': df['sockets'].min(),
    'sockets_max': df['sockets'].max(),
    'rate_min': df['rate'].min(),
    'rate_max': df['rate'].max(),
    'missing_neg_pct': (df['missing'] < 0).mean(),
    'missing_gt_sockets_pct': (df['missing'] > df['sockets']).mean(),
}

print('SUMMARY', summary)

# Simple group means for rate
print('\nRate by genus (mean, sd, n):')
print(df.groupby('genus')['rate'].agg(['mean','std','count']))

# OLS regression: rate ~ is_human + age + sex + tooth_class
model = smf.ols('rate ~ is_human + age + sex + C(tooth_class)', data=df).fit(cov_type='HC3')
print('\nOLS rate ~ is_human + age + sex + tooth_class (HC3)')
print(model.summary())

# Also run with genus categorical (Homo vs others) to see consistency
model_genus = smf.ols('rate ~ C(genus) + age + sex + C(tooth_class)', data=df).fit(cov_type='HC3')
print('\nOLS rate ~ genus + age + sex + tooth_class (HC3)')
print(model_genus.summary())

# Contrast: Homo sapiens vs average of non-human genera
# Using the genus model
# Build contrast vector for parameters order
params = model_genus.params.index.tolist()

# Identify parameter names for genera
# Baseline is first alphabetically by statsmodels (likely Homo sapiens?)
print('\nParam order:', params)

# Use design matrix to compute estimated marginal means by genus
# Create a dataframe with average age/sex and each genus, and average tooth_class distribution? We will use reference values.

# Use average values and most common tooth_class to get direction (not strict marginal mean)
ref_age = df['age'].mean()
ref_sex = df['sex'].mean()
ref_tooth = df['tooth_class'].mode()[0]

rows = []
for g in df['genus'].unique():
    rows.append({'genus': g, 'age': ref_age, 'sex': ref_sex, 'tooth_class': ref_tooth})
ref_df = pd.DataFrame(rows)

pred = model_genus.get_prediction(ref_df)
ref_df['pred_rate'] = pred.predicted_mean
print('\nReference predictions (same covariates):')
print(ref_df)
