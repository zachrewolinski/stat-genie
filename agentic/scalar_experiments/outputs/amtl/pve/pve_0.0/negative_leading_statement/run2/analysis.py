import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Basic cleaning: drop rows with missing key variables
cols = ['num_amtl','age','prob_male','tooth_class','genus']
_df = _df.dropna(subset=cols).copy()

# Ensure categories
_df['tooth_class'] = _df['tooth_class'].astype('category')
# Set genus category order with Homo sapiens as reference if present
if 'Homo sapiens' in _df['genus'].unique():
    genus_order = ['Homo sapiens'] + [g for g in sorted(_df['genus'].unique()) if g != 'Homo sapiens']
    _df['genus'] = pd.Categorical(_df['genus'], categories=genus_order, ordered=False)

# Binary human indicator
_df['is_human'] = (_df['genus'] == 'Homo sapiens').astype(int)

# Model 1: binary human vs non-human
m1 = smf.ols('num_amtl ~ is_human + age + prob_male + C(tooth_class)', data=_df).fit(cov_type='HC3')

# Model 2: genus categorical to compare each genus vs Homo sapiens
m2 = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=_df).fit(cov_type='HC3')

# Summaries
out = {}

out['n'] = int(_df.shape[0])

# Coef for is_human
coef_h = m1.params.get('is_human', np.nan)
se_h = m1.bse.get('is_human', np.nan)
p_h = m1.pvalues.get('is_human', np.nan)

out['m1_is_human_coef'] = float(coef_h)
out['m1_is_human_se'] = float(se_h)
out['m1_is_human_p'] = float(p_h)

# Means for context (unadjusted)
means = _df.groupby('genus')['num_amtl'].mean().to_dict()
out['unadjusted_means'] = {k: float(v) for k, v in means.items()}

# Genus comparisons vs Homo
genus_coefs = {}
for term in m2.params.index:
    if term.startswith('C(genus)'):
        genus_coefs[term] = {
            'coef': float(m2.params[term]),
            'se': float(m2.bse[term]),
            'p': float(m2.pvalues[term])
        }

out['m2_genus_terms'] = genus_coefs

# Write a small text summary for debugging (not required by instructions, but helpful in logs)
print('N:', out['n'])
print('is_human coef (HC3):', out['m1_is_human_coef'], 'se', out['m1_is_human_se'], 'p', out['m1_is_human_p'])
print('Unadjusted means:', out['unadjusted_means'])
print('Genus terms vs Homo (HC3):')
for k,v in out['m2_genus_terms'].items():
    print(' ', k, v)

# Save analysis outputs for later use
pd.Series(out).to_json('analysis_results.json')
