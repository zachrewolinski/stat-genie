import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Basic cleaning
_df = _df.copy()
_df['human'] = (_df['genus'] == 'Homo sapiens').astype(int)

# OLS with cluster-robust SE by specimen to account for repeated measures
model = smf.ols('num_amtl ~ human + age + prob_male + C(tooth_class)', data=_df).fit(
    cov_type='cluster', cov_kwds={'groups': _df['specimen']}
)

# Extract key results
coef = model.params['human']
se = model.bse['human']
pval = model.pvalues['human']

# Compute effect size relative to outcome SD
outcome_sd = _df['num_amtl'].std(ddof=0)
std_effect = coef / outcome_sd if outcome_sd != 0 else np.nan

# Also fit genus categorical model for comparison
model_genus = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=_df).fit(
    cov_type='cluster', cov_kwds={'groups': _df['specimen']}
)

# Extract contrasts vs baseline genus (alphabetical by default)
# Determine baseline
baseline = sorted(_df['genus'].unique())[0]

results = {
    'n_rows': int(_df.shape[0]),
    'n_specimens': int(_df['specimen'].nunique()),
    'human_coef': float(coef),
    'human_se': float(se),
    'human_pval': float(pval),
    'human_std_effect': float(std_effect),
    'baseline_genus': baseline,
    'genus_params': {k: float(v) for k, v in model_genus.params.items() if k.startswith('C(genus)')},
    'genus_pvalues': {k: float(v) for k, v in model_genus.pvalues.items() if k.startswith('C(genus)')},
}

print(results)
