import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
DF_PATH = 'amtl.csv'
df = pd.read_csv(DF_PATH)

# Create indicator for humans vs non-humans
# Ensure consistent casing/spacing
is_human = df['genus'].astype(str).str.strip().eq('Homo sapiens')
df = df.copy()
df['is_human'] = is_human.astype(int)

# Keep relevant variables
cols = ['num_amtl', 'age', 'prob_male', 'tooth_class', 'genus', 'is_human']
analysis_df = df[cols].dropna()

# OLS model: num_amtl ~ is_human + age + prob_male + tooth_class
model = smf.ols('num_amtl ~ is_human + age + prob_male + C(tooth_class)', data=analysis_df).fit(cov_type='HC3')

# Extract key stats
coef = model.params['is_human']
se = model.bse['is_human']
pval = model.pvalues['is_human']

# Also fit full genus model for pairwise differences
model_genus = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=analysis_df).fit(cov_type='HC3')

# Compute adjusted mean difference between Homo sapiens and each non-human genus
# Use linear hypothesis with model_genus
params = model_genus.params
cov = model_genus.cov_params()

# Reference category is first alphabetically by default in statsmodels: 'Homo sapiens' maybe? actually C(genus)[T.Pan] etc.
# We'll compute predicted difference Homo - other using model design.
# If Homo sapiens is reference, then coefficients for others are differences vs Homo.
# If not, compute manually via releveling.

# Determine reference
levels = sorted(analysis_df['genus'].unique())
# Statsmodels uses alphabetical ordering for categorical by default
ref = levels[0]

pairwise = {}

if ref == 'Homo sapiens':
    # Coef for other genus is (other - Homo)
    for g in levels:
        if g == 'Homo sapiens':
            continue
        term = f'C(genus)[T.{g}]'
        if term in params:
            diff = -params[term]  # Homo - other
            se_diff = np.sqrt(cov.loc[term, term])
            # two-sided p for difference
            t = diff / se_diff if se_diff > 0 else np.nan
            # approximate p via normal
            from scipy import stats
            p = 2 * (1 - stats.t.cdf(abs(t), df=model_genus.df_resid))
        else:
            diff = np.nan
            se_diff = np.nan
            p = np.nan
        pairwise[g] = {'diff_homo_minus_other': diff, 'se': se_diff, 'pval': p}
else:
    # Refit with Homo sapiens as reference
    model_genus_ref = smf.ols('num_amtl ~ C(genus, Treatment(reference="Homo sapiens")) + age + prob_male + C(tooth_class)', data=analysis_df).fit(cov_type='HC3')
    params2 = model_genus_ref.params
    cov2 = model_genus_ref.cov_params()
    for g in levels:
        if g == 'Homo sapiens':
            continue
        term = f'C(genus, Treatment(reference="Homo sapiens"))[T.{g}]'
        if term in params2:
            diff = -params2[term]
            se_diff = np.sqrt(cov2.loc[term, term])
            from scipy import stats
            t = diff / se_diff if se_diff > 0 else np.nan
            p = 2 * (1 - stats.t.cdf(abs(t), df=model_genus_ref.df_resid))
        else:
            diff = np.nan
            se_diff = np.nan
            p = np.nan
        pairwise[g] = {'diff_homo_minus_other': diff, 'se': se_diff, 'pval': p}

# Compute adjusted mean difference between Homo and average of non-humans (linear contrast)
# Use model with is_human already as above for primary decision

results = {
    'n': int(analysis_df.shape[0]),
    'is_human_coef': float(coef),
    'is_human_se': float(se),
    'is_human_pval': float(pval),
    'pairwise': pairwise,
    'ref_genus': ref,
    'model_r2': float(model.rsquared)
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
