import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Basic cleaning
_df = _df.dropna()

# Derived variables
_df['is_human'] = (_df['feature8'] == 'Homo sapiens').astype(int)

# Summary info
n_rows = len(_df)
unique_specimens = _df['feature2'].nunique()

# OLS: human vs non-human, controlling for age, sex, tooth class
model_binary = smf.ols(
    'feature3 ~ is_human + feature5 + feature7 + C(feature1)',
    data=_df
).fit(cov_type='cluster', cov_kwds={'groups': _df['feature2']})

coef_human = model_binary.params['is_human']
se_human = model_binary.bse['is_human']
p_human = model_binary.pvalues['is_human']
ci_low, ci_high = model_binary.conf_int().loc['is_human']

# Standardized effect size (Cohen's d using outcome SD)
outcome_sd = _df['feature3'].std(ddof=1)
cohen_d = coef_human / outcome_sd if outcome_sd else np.nan

# Adjusted mean predictions for human vs non-human
_df_human = _df.copy()
_df_human['is_human'] = 1
_df_non = _df.copy()
_df_non['is_human'] = 0
pred_human = model_binary.predict(_df_human).mean()
pred_non = model_binary.predict(_df_non).mean()

# Genus-specific model with Homo sapiens as reference
model_genus = smf.ols(
    'feature3 ~ C(feature8, Treatment(reference="Homo sapiens")) + feature5 + feature7 + C(feature1)',
    data=_df
).fit(cov_type='cluster', cov_kwds={'groups': _df['feature2']})

# Extract genus effects
params = model_genus.params
bse = model_genus.bse
pvals = model_genus.pvalues
conf = model_genus.conf_int()

genus_effects = {}
for term in params.index:
    if term.startswith('C(feature8'):
        genus = term.split(']')[-1]  # not reliable

# Better: map terms to genus names
for term in params.index:
    if term.startswith('C(feature8'):
        # term looks like C(feature8, Treatment(reference="Homo sapiens"))[T.Pan]
        genus = term.split('[T.')[-1].rstrip(']')
        genus_effects[genus] = {
            'coef': params[term],
            'se': bse[term],
            'p': pvals[term],
            'ci_low': conf.loc[term, 0],
            'ci_high': conf.loc[term, 1],
        }

# Adjusted means for each genus using genus-specific model
adjusted_means = {}
for genus in _df['feature8'].unique():
    _tmp = _df.copy()
    _tmp['feature8'] = genus
    adjusted_means[genus] = model_genus.predict(_tmp).mean()

# Print results
print(f"Rows: {n_rows}, specimens: {unique_specimens}")
print("Genus counts:")
print(_df['feature8'].value_counts())
print("\nBinary human vs non-human model:")
print(f"  coef(is_human) = {coef_human:.4f}")
print(f"  SE = {se_human:.4f}, p = {p_human:.4g}")
print(f"  95% CI = [{ci_low:.4f}, {ci_high:.4f}]")
print(f"  Cohen's d (approx) = {cohen_d:.3f}")
print(f"  Adjusted mean (human) = {pred_human:.4f}")
print(f"  Adjusted mean (non-human) = {pred_non:.4f}")
print(f"  Adjusted mean difference = {pred_human - pred_non:.4f}")

print("\nGenus-specific model (reference: Homo sapiens):")
for genus, stats in genus_effects.items():
    print(f"  {genus}: coef = {stats['coef']:.4f}, SE = {stats['se']:.4f}, p = {stats['p']:.4g}, CI = [{stats['ci_low']:.4f}, {stats['ci_high']:.4f}]")

print("\nAdjusted means by genus:")
for genus, mean_val in adjusted_means.items():
    print(f"  {genus}: {mean_val:.4f}")
