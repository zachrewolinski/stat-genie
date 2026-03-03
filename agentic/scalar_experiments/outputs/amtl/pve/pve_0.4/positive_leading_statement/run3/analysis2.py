import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data

df = pd.read_csv('amtl.csv')

# Quick summaries
print(df.groupby('genus')['num_amtl'].agg(['mean','std','count']))

# Linear model with covariates
model = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=df).fit()
print(model.summary())

# Use robust SE?
robust = model.get_robustcov_results(cov_type='HC3')
print(robust.summary())

# Get estimated marginal means for genus vs ref (Homo sapiens baseline?)
# Baseline is alphabetically? C(genus) uses first alphabetically by default (Homo sapiens, Pan, Papio, Pongo) because sorts? Let's check.
print('Genus categories:', df['genus'].unique())

# Use Homo sapiens as reference explicitly
model2 = smf.ols('num_amtl ~ C(genus, Treatment(reference="Homo sapiens")) + age + prob_male + C(tooth_class)', data=df).fit(cov_type='HC3')
print(model2.summary())

# compute pairwise difference of Homo sapiens vs each genus
params = model2.params
conf = model2.conf_int()
print('params:', params)
print('conf:', conf)

