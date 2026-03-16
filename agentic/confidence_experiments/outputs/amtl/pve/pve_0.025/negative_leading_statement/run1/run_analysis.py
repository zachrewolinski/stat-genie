import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
path = 'amtl.csv'
df = pd.read_csv(path)

# Create human indicator
# Non-human genera: Pan, Pongo, Papio
# Human: Homo sapiens

df['human'] = (df['genus'] == 'Homo sapiens').astype(int)

# Center age to reduce collinearity if we use polynomial; keep simple linear term
# We'll use age as is, plus prob_male and tooth_class

# Fit OLS with robust SEs
model = smf.ols('num_amtl ~ human + age + prob_male + C(tooth_class)', data=df).fit(cov_type='HC3')

# Effect size: standardized by outcome SD
outcome_sd = df['num_amtl'].std(ddof=1)
coef = model.params['human']
se = model.bse['human']
pval = model.pvalues['human']
ci_low, ci_high = model.conf_int().loc['human']

# Standardized effect size (Cohen's d style) for continuous outcome
std_effect = coef / outcome_sd

# Also compare using full genus categories to see Homo vs others individually
model_genus = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=df).fit(cov_type='HC3')

# Extract coefficients for genus (reference is first category alphabetically?) Statsmodels chooses alphabetical: 'Homo sapiens' maybe first? Let's check categories.
# We'll compute predicted difference between Homo and each non-human genus using model with genus.

# Get category order
cat_order = sorted(df['genus'].unique())

# Build predictions for a reference covariate profile
ref = {
    'age': df['age'].mean(),
    'prob_male': df['prob_male'].mean(),
    'tooth_class': df['tooth_class'].mode()[0],
    'genus': None,
}

preds = {}
for g in ['Homo sapiens','Pan','Pongo','Papio']:
    ref['genus'] = g
    preds[g] = float(model_genus.predict(pd.DataFrame([ref]))[0])

# Pairwise differences Homo - others
pairwise = {g: preds['Homo sapiens'] - preds[g] for g in ['Pan','Pongo','Papio']}

# Output summary
print('human coef', coef)
print('human se', se)
print('human pval', pval)
print('human 95% CI', (ci_low, ci_high))
print('standardized effect', std_effect)
print('n', len(df))
print('R2', model.rsquared)
print('pairwise predicted differences (Homo - other):', pairwise)

# Save results for later use
results = {
    'coef': coef,
    'se': se,
    'pval': pval,
    'ci_low': ci_low,
    'ci_high': ci_high,
    'std_effect': std_effect,
    'pairwise': pairwise,
    'r2': model.rsquared,
    'n': len(df),
}

import json
with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)
