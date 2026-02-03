import pandas as pd
import numpy as np
import statsmodels.api as sm
import patsy as pt

# Load data
_df = pd.read_csv('amtl.csv')

# Basic cleanup: ensure categories
_df['genus'] = _df['genus'].astype('category')
_df['tooth_class'] = _df['tooth_class'].astype('category')

# Binomial response: successes=num_amtl, failures=sockets-num_amtl
# Use GLM with logit link
# Reference categories will be alphabetical by default; ensure Homo sapiens is explicit in results

_df['num_present'] = _df['sockets'] - _df['num_amtl']
formula = 'num_amtl + num_present ~ C(genus) + age + prob_male + C(tooth_class)'

y, X = pt.dmatrices(formula, data=_df, return_type='dataframe')
model = sm.GLM(y, X, family=sm.families.Binomial()).fit()

print(model.summary())

# Extract genus coefficients vs reference (alphabetical)
# We want to know if Homo sapiens has higher AMTL than non-human genera.
# We'll refit with Homo sapiens as reference to directly compare others to Homo.

_df['genus'] = _df['genus'].cat.reorder_categories(
    ['Homo sapiens', 'Pan', 'Papio', 'Pongo'], ordered=False
)

y2, X2 = pt.dmatrices(formula, data=_df, return_type='dataframe')
model_homo_ref = sm.GLM(y2, X2, family=sm.families.Binomial()).fit()

print("\nModel with Homo sapiens as reference:")
print(model_homo_ref.summary())

# Compute odds ratios for non-human genera vs Homo
params = model_homo_ref.params
conf = model_homo_ref.conf_int()

rows = []
for term in params.index:
    if term.startswith('C(genus)'):
        or_val = float(pd.Series(params[term]).apply(lambda x: np.exp(x)))
        ci_low = float(pd.Series(conf.loc[term, 0]).apply(lambda x: np.exp(x)))
        ci_high = float(pd.Series(conf.loc[term, 1]).apply(lambda x: np.exp(x)))
        pval = float(model_homo_ref.pvalues[term])
        rows.append((term, or_val, ci_low, ci_high, pval))

print("\nOdds ratios (non-human vs Homo sapiens reference):")
for term, or_val, ci_low, ci_high, pval in rows:
    print(f"{term}: OR={or_val:.3f}, 95% CI [{ci_low:.3f}, {ci_high:.3f}], p={pval:.4g}")
