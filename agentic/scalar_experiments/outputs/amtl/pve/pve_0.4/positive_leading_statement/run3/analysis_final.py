import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('amtl.csv')

# Fit linear model with Homo sapiens as reference and robust SE
formula = 'num_amtl ~ C(genus, Treatment(reference="Homo sapiens")) + age + prob_male + C(tooth_class)'
model = smf.ols(formula, data=df).fit(cov_type='HC3')

# Extract genus coefficients (differences vs Homo sapiens)
coef = model.params
pvals = model.pvalues

# Compute adjusted (marginal) means by setting genus to each level and averaging predictions
levels = ['Homo sapiens', 'Pan', 'Papio', 'Pongo']
means = {}
for g in levels:
    tmp = df.copy()
    tmp['genus'] = g
    means[g] = model.predict(tmp).mean()

# Compute differences vs Homo sapiens for adjusted means
base = means['Homo sapiens']
diffs = {g: base - means[g] for g in levels if g != 'Homo sapiens'}

# Print outputs for reference
print('Adjusted means (model-based):', means)
print('Differences vs Homo sapiens (positive means Homo higher):', diffs)
print('Coefficients vs Homo sapiens:', {k:v for k,v in coef.items() if 'C(genus' in k})
print('P-values for genus terms:', {k:v for k,v in pvals.items() if 'C(genus' in k})

