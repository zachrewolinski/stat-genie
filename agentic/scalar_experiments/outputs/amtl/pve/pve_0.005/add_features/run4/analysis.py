import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
DF_PATH = 'amtl.csv'

df = pd.read_csv(DF_PATH)

# Keep relevant columns
cols = ['num_amtl', 'genus', 'age', 'prob_male', 'tooth_class', 'specimen', 'sockets']
missing_cols = [c for c in cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")

# Drop missing in relevant columns
use_df = df[cols].dropna().copy()

# Set category ordering for consistent reference levels
use_df['genus'] = pd.Categorical(
    use_df['genus'],
    categories=['Pan', 'Pongo', 'Papio', 'Homo sapiens'],
    ordered=True,
)
use_df['tooth_class'] = pd.Categorical(
    use_df['tooth_class'],
    categories=['Anterior', 'Premolar', 'Posterior'],
    ordered=True,
)

# OLS with cluster-robust SEs by specimen
model = smf.ols(
    'num_amtl ~ C(genus) + age + prob_male + C(tooth_class)',
    data=use_df,
).fit(cov_type='cluster', cov_kwds={'groups': use_df['specimen']})

print(model.summary())

# Extract coefficients
params = model.params
bse = model.bse
pvalues = model.pvalues

# Helper for t-tests on contrasts
# statsmodels order of params is in model.params index
param_names = list(params.index)

# Build contrast vectors
# Baseline genus is Pan. Coefs: C(genus)[T.Pongo], C(genus)[T.Papio], C(genus)[T.Homo sapiens]

def contrast_test(lhs, rhs):
    """Test lhs - rhs = 0, where lhs/rhs are coef names. rhs can be None for baseline."""
    # Build contrast vector
    c = np.zeros(len(param_names))
    if lhs is not None:
        c[param_names.index(lhs)] = 1.0
    if rhs is not None:
        c[param_names.index(rhs)] -= 1.0
    res = model.t_test(c)
    return {
        'estimate': float(res.effect),
        'se': float(res.sd),
        't': float(res.tvalue),
        'p': float(res.pvalue),
    }

# Comparisons: Homo vs Pan (baseline), Homo vs Pongo, Homo vs Papio
homo = 'C(genus)[T.Homo sapiens]'
pongo = 'C(genus)[T.Pongo]'
papio = 'C(genus)[T.Papio]'

comparisons = {
    'Homo_vs_Pan': contrast_test(homo, None),
    'Homo_vs_Pongo': contrast_test(homo, pongo),
    'Homo_vs_Papio': contrast_test(homo, papio),
}

print('\nAdjusted differences (Homo sapiens vs others):')
for k, v in comparisons.items():
    print(k, v)

# Binary human vs nonhuman model
use_df['human'] = (use_df['genus'] == 'Homo sapiens').astype(int)
model2 = smf.ols(
    'num_amtl ~ human + age + prob_male + C(tooth_class)',
    data=use_df,
).fit(cov_type='cluster', cov_kwds={'groups': use_df['specimen']})

print('\nBinary human vs nonhuman model:')
print(model2.summary())
print('human coef:', model2.params['human'], 'p=', model2.pvalues['human'])

