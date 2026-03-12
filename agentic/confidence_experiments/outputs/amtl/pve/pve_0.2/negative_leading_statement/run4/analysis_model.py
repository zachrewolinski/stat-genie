import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Set genus categories with Homo sapiens as reference
cat_order = ["Homo sapiens", "Pan", "Pongo", "Papio"]
_df['genus'] = pd.Categorical(_df['genus'], categories=cat_order, ordered=False)

# Basic descriptive stats
means_by_genus = _df.groupby('genus')['num_amtl'].mean()
counts_by_genus = _df['genus'].value_counts().sort_index()

# Fit OLS with categorical predictors
formula = 'num_amtl ~ C(genus, Treatment(reference="Homo sapiens")) + age + prob_male + C(tooth_class)'
model = smf.ols(formula, data=_df).fit()

# Cluster-robust SEs by specimen (repeated measures across tooth_class)
robust = model.get_robustcov_results(cov_type='cluster', groups=_df['specimen'])

# Extract coefficients for genus comparisons (non-human vs human)
params = np.asarray(robust.params)
pvalues = np.asarray(robust.pvalues)
param_names = list(robust.model.exog_names)

# Contrast: average non-human difference vs Homo sapiens
contrast = np.zeros(len(param_names))
for g in ["C(genus, Treatment(reference=\"Homo sapiens\"))[T.Pan]",
          "C(genus, Treatment(reference=\"Homo sapiens\"))[T.Pongo]",
          "C(genus, Treatment(reference=\"Homo sapiens\"))[T.Papio]"]:
    if g in param_names:
        contrast[param_names.index(g)] = 1/3

avg_nonhuman_test = robust.t_test(contrast)

# Save results to stdout
print('Means by genus (raw):')
print(means_by_genus)
print('\nCounts by genus:')
print(counts_by_genus)

print('\nGenus coefficients (non-human vs Homo sapiens) with cluster-robust SEs:')
for g in ["C(genus, Treatment(reference=\"Homo sapiens\"))[T.Pan]",
          "C(genus, Treatment(reference=\"Homo sapiens\"))[T.Pongo]",
          "C(genus, Treatment(reference=\"Homo sapiens\"))[T.Papio]"]:
    if g in param_names:
        idx = param_names.index(g)
        print(g, 'coef=', float(params[idx]), 'p=', float(pvalues[idx]))

print('\nAverage non-human minus Homo sapiens contrast:')
print('estimate=', float(avg_nonhuman_test.effect), 'p=', float(avg_nonhuman_test.pvalue))

# Also provide model R-squared for context
print('\nModel R-squared:', model.rsquared)
