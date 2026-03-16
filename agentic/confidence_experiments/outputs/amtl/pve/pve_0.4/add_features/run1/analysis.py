import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
path = 'amtl.csv'
df = pd.read_csv(path)

# Keep relevant columns and drop missing
cols = ['num_amtl','genus','age','prob_male','tooth_class','specimen']
df = df[cols].dropna()

# Ensure categories
# Set Homo sapiens as reference
if 'Homo sapiens' in df['genus'].unique():
    genus_cat = ['Homo sapiens'] + [g for g in sorted(df['genus'].unique()) if g != 'Homo sapiens']
    df['genus'] = pd.Categorical(df['genus'], categories=genus_cat)
else:
    df['genus'] = df['genus'].astype('category')

# Tooth class as categorical
if 'tooth_class' in df.columns:
    df['tooth_class'] = df['tooth_class'].astype('category')

# Fit OLS with cluster-robust SE by specimen
formula = "num_amtl ~ C(genus, Treatment(reference='Homo sapiens')) + age + prob_male + C(tooth_class)"
model = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['specimen']})

# Extract genus effects
params = model.params
pvalues = model.pvalues

# Identify genus coefficient names
coef_names = [name for name in params.index if name.startswith('C(genus')]

# Build contrast: Homo sapiens vs average of non-human genera
# With Homo as reference, difference Homo - avg(nonhuman) = -avg(beta_nonhuman)
if coef_names:
    # contrast vector for t_test
    L = np.zeros(len(params))
    # average of non-human coefficients
    nonhuman_coefs = []
    for name in coef_names:
        idx = params.index.get_loc(name)
        L[idx] = -1.0 / len(coef_names)
        nonhuman_coefs.append(name)
    # t-test
    t_res = model.t_test(L)
    avg_diff = float(t_res.effect)
    avg_diff_se = float(t_res.sd)
    avg_diff_p = float(t_res.pvalue)
else:
    avg_diff = avg_diff_se = avg_diff_p = np.nan

# Summaries for reporting
summary = {
    'n': int(df.shape[0]),
    'genus_levels': df['genus'].cat.categories.tolist() if hasattr(df['genus'], 'cat') else sorted(df['genus'].unique()),
    'coef_table': {name: {'coef': float(params[name]), 'p': float(pvalues[name])} for name in coef_names},
    'avg_homo_minus_nonhuman': {'diff': avg_diff, 'se': avg_diff_se, 'p': avg_diff_p}
}

print(summary)
