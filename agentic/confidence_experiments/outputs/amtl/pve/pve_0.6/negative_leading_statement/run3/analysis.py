import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
import json

# Load data
_df = pd.read_csv('amtl.csv')

# Create indicator for human
_df['is_human'] = (_df['genus'] == 'Homo sapiens').astype(int)

# Base model: human vs non-human, controlling for age, sex, tooth class
model = smf.ols('num_amtl ~ is_human + age + prob_male + C(tooth_class)', data=_df).fit(
    cov_type='cluster', cov_kwds={'groups': _df['specimen']}
)

# Genus-specific model (Homo sapiens as reference), to compare each non-human genus
model_genus = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=_df).fit(
    cov_type='cluster', cov_kwds={'groups': _df['specimen']}
)

# Extract key stats
coef = model.params['is_human']
se = model.bse['is_human']
pval = model.pvalues['is_human']

# Collect genus coefficients (relative to Homo sapiens)
# Non-human genus coefficients indicate difference vs Homo sapiens
nonhuman_effects = {}
for term in model_genus.params.index:
    if term.startswith('C(genus)'):
        nonhuman_effects[term] = {
            'coef': float(model_genus.params[term]),
            'se': float(model_genus.bse[term]),
            'pval': float(model_genus.pvalues[term]),
        }

summary = {
    'is_human_coef': float(coef),
    'is_human_se': float(se),
    'is_human_pval': float(pval),
    'nonhuman_vs_human': nonhuman_effects,
}

with open('analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
