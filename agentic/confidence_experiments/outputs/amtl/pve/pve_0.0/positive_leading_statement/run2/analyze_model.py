import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import patsy

# Load data
path = 'amtl.csv'

df = pd.read_csv(path)

# Ensure categorical types
for col in ['genus','tooth_class']:
    df[col] = df[col].astype('category')

# Set reference categories
# We want Homo sapiens as reference for genus, and Anterior for tooth_class
if 'Homo sapiens' in df['genus'].cat.categories:
    df['genus'] = df['genus'].cat.reorder_categories(
        ['Homo sapiens'] + [g for g in df['genus'].cat.categories if g != 'Homo sapiens'],
        ordered=False
    )

if 'Anterior' in df['tooth_class'].cat.categories:
    df['tooth_class'] = df['tooth_class'].cat.reorder_categories(
        ['Anterior'] + [c for c in df['tooth_class'].cat.categories if c != 'Anterior'],
        ordered=False
    )

# Fit OLS model with robust SEs
formula = 'num_amtl ~ C(genus) + age + prob_male + C(tooth_class)'
model = smf.ols(formula, data=df).fit(cov_type='HC3')

print(model.summary())

# Extract coefficients for genus (non-human relative to Homo sapiens)
coef = model.params
se = model.bse
pvals = model.pvalues

# Identify genus terms
for term in coef.index:
    if term.startswith('C(genus)'):
        print(term, coef[term], se[term], pvals[term])

# Compute adjusted means (marginal means) for each genus
# Standardize over observed covariate distribution for age, prob_male, tooth_class.
levels = df['genus'].cat.categories.tolist()

pred_means = {}
for level in levels:
    df_tmp = df.copy()
    df_tmp['genus'] = level
    pred = model.predict(df_tmp)
    pred_means[level] = pred.mean()

print('Adjusted means (standardized over covariates):')
for level, mean in pred_means.items():
    print(level, mean)

# Compute pairwise differences: Homo sapiens vs each non-human using design matrix

y, X = patsy.dmatrices(formula, data=df, return_type='dataframe')

# Function to compute mean design vector for a given genus

def mean_design_for_genus(level):
    df_tmp = df.copy()
    df_tmp['genus'] = level
    _, X_tmp = patsy.dmatrices(formula, data=df_tmp, return_type='dataframe')
    return X_tmp.mean(axis=0)

means_X = {level: mean_design_for_genus(level) for level in levels}

# covariance matrix of coefficients
cov = model.cov_params()

# Helper for p-value from normal approximation

def normal_pvalue(z):
    return 2 * (1 - sm.stats.norm.cdf(abs(z)))

for level in levels:
    if level == 'Homo sapiens':
        continue
    diff = (means_X['Homo sapiens'] - means_X[level]).reindex(coef.index)
    diff_val = float(diff.values @ coef.values)
    diff_se = float(np.sqrt(diff.values @ cov.values @ diff.values))
    z = diff_val / diff_se if diff_se > 0 else np.nan
    p = normal_pvalue(z) if diff_se > 0 else np.nan
    print(f'Difference Homo sapiens - {level}: {diff_val:.4f} (SE {diff_se:.4f}), z={z:.3f}, p={p:.4g}')

# Combined non-human mean (equal weight across genera)
non_human_levels = [lvl for lvl in levels if lvl != 'Homo sapiens']
mean_nonhuman_X = sum(means_X[lvl] for lvl in non_human_levels) / len(non_human_levels)

diff = (means_X['Homo sapiens'] - mean_nonhuman_X).reindex(coef.index)

diff_val = float(diff.values @ coef.values)
diff_se = float(np.sqrt(diff.values @ cov.values @ diff.values))
z = diff_val / diff_se if diff_se > 0 else np.nan
p = normal_pvalue(z) if diff_se > 0 else np.nan
print(f'Difference Homo sapiens - mean(non-human): {diff_val:.4f} (SE {diff_se:.4f}), z={z:.3f}, p={p:.4g}')

