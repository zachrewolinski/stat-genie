import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats

# Load data
path = 'amtl.csv'
df = pd.read_csv(path)

# Basic cleaning: drop rows with missing key fields
key_cols = ['num_amtl', 'age', 'prob_male', 'genus', 'tooth_class']
df_model = df.dropna(subset=key_cols).copy()

# Fit linear model with genus + age + sex + tooth_class
model = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=df_model).fit(cov_type='HC3')

# Determine reference category in statsmodels for genus
all_genus_levels = sorted(df_model['genus'].dropna().unique())
ref_genus = all_genus_levels[0]

params = model.params
cov = model.cov_params()
param_index = list(params.index)

# Helper to get coefficient for a genus (reference has 0)
def coef_for(genus):
    if genus == ref_genus:
        return 0.0
    col = f'C(genus)[T.{genus}]'
    return params.get(col, 0.0)

# Pairwise comparisons Homo sapiens vs each non-human genus
comparisons = {}
for g in all_genus_levels:
    if g == 'Homo sapiens':
        continue
    diff = coef_for('Homo sapiens') - coef_for(g)
    contrast = np.zeros(len(params))

    def add_to_contrast(genus, sign):
        if genus == ref_genus:
            return
        col = f'C(genus)[T.{genus}]'
        if col in param_index:
            contrast[param_index.index(col)] += sign

    add_to_contrast('Homo sapiens', 1.0)
    add_to_contrast(g, -1.0)

    var = contrast @ cov.values @ contrast
    se = np.sqrt(var) if var >= 0 else np.nan
    t = diff / se if se and se > 0 else np.nan
    p = 2 * (1 - stats.norm.cdf(abs(t))) if t == t else np.nan

    comparisons[g] = {'diff': diff, 'se': se, 't': t, 'p': p}

# Overall effect of genus (type II ANOVA)
anova = sm.stats.anova_lm(model, typ=2)

# Least-squares means by averaging predicted values over observed covariates
lsmeans = {}
for g in all_genus_levels:
    df_pred = df_model.copy()
    df_pred['genus'] = g
    pred = model.predict(df_pred)
    lsmeans[g] = pred.mean()

# Contrast: Homo sapiens vs average of non-human genera (Pan, Papio, Pongo)
nonhuman = [g for g in all_genus_levels if g != 'Homo sapiens']
weights = {g: 1 / len(nonhuman) for g in nonhuman}

# Build contrast vector for parameters
contrast = np.zeros(len(params))

def add_to_contrast(genus, weight):
    if genus == ref_genus:
        # reference coefficient is implicit 0, so nothing to add
        return
    col = f'C(genus)[T.{genus}]'
    if col in param_index:
        contrast[param_index.index(col)] += weight

# Homo sapiens minus average of non-human
add_to_contrast('Homo sapiens', 1.0)
for g in nonhuman:
    add_to_contrast(g, -weights[g])

# Compute contrast estimate
# Estimate for Homo sapiens is 0 if it is reference; otherwise its coefficient
estimate = coef_for('Homo sapiens') - sum(weights[g] * coef_for(g) for g in nonhuman)
var = contrast @ cov.values @ contrast
se = np.sqrt(var) if var >= 0 else np.nan
t = estimate / se if se and se > 0 else np.nan
p = 2 * (1 - stats.norm.cdf(abs(t))) if t == t else np.nan

print('Reference genus:', ref_genus)
print('Genus levels:', all_genus_levels)
print('N:', len(df_model))
print('\nModel summary (genus coefficients):')
for name, val in params.items():
    if name.startswith('C(genus)'):
        print(name, val)

print('\nPairwise comparisons Homo sapiens vs non-human:')
for g, stats_ in comparisons.items():
    print(g, stats_)

print('\nANOVA (type II) for genus:')
print(anova.loc[['C(genus)']])

print('\nLSmeans:')
for g, m in lsmeans.items():
    print(g, m)

print('\nContrast Homo sapiens vs avg non-human:')
print({'estimate': estimate, 'se': se, 't': t, 'p': p})

