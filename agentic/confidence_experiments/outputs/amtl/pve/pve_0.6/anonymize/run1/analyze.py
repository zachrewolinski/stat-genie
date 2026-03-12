import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

df = _df.copy()
# Variables
# feature1: tooth class (Anterior/Posterior/Premolar)
# feature3: AMTL measure (continuous, anonymized)
# feature5: age at death
# feature7: sex estimate (0-1 scale)
# feature8: genus

# Binary indicator for Homo sapiens
_df['human'] = (_df['feature8'] == 'Homo sapiens').astype(int)

# Model 1: human vs non-human, controlling for age, sex, tooth class
model1 = smf.ols('feature3 ~ human + feature5 + feature7 + C(feature1)', data=_df).fit(cov_type='HC3')

# Model 2: genus categories to compare Homo sapiens vs each non-human genus
model2 = smf.ols('feature3 ~ C(feature8) + feature5 + feature7 + C(feature1)', data=_df).fit(cov_type='HC3')

# Helper: adjusted mean by g-computation
covars = ['feature5', 'feature7', 'feature1']
base = _df[covars].copy()

def adjusted_mean_for_genus(genus: str) -> float:
    tmp = base.copy()
    tmp['feature8'] = genus
    preds = model2.predict(tmp)
    return float(preds.mean())

adj_means = {g: adjusted_mean_for_genus(g) for g in sorted(_df['feature8'].unique())}

# Pairwise contrasts: Homo sapiens vs other genera
params = model2.params
cov = model2.cov_params()

# Build contrast vectors for Homo sapiens vs each genus
# statsmodels uses treatment coding with baseline alphabetically (by default)
param_names = list(params.index)

# Function to build contrast for difference in predicted mean between two genera
# Since model is linear, difference is just difference in their genus coefficients.
# We compute contrast in parameter space.

def contrast_vector(genus_a: str, genus_b: str) -> np.ndarray:
    # baseline is the first level in sorted categories
    levels = sorted(_df['feature8'].unique())
    baseline = levels[0]
    vec = np.zeros(len(param_names))

    # Intercept cancels
    # For genus coding, params look like C(feature8)[T.Gen]
    def add_genus_effect(genus: str, sign: float):
        if genus == baseline:
            return
        name = f'C(feature8)[T.{genus}]'
        if name in param_names:
            vec[param_names.index(name)] += sign

    add_genus_effect(genus_a, 1.0)
    add_genus_effect(genus_b, -1.0)
    return vec

contrast_results = {}
for genus in sorted(_df['feature8'].unique()):
    if genus == 'Homo sapiens':
        continue
    v = contrast_vector('Homo sapiens', genus)
    est = float(np.dot(v, params))
    se = float(np.sqrt(np.dot(v, np.dot(cov, v))))
    t = est / se if se > 0 else np.nan
    # two-sided p-value
    from scipy import stats
    p = float(2 * (1 - stats.t.cdf(abs(t), df=model2.df_resid))) if np.isfinite(t) else np.nan
    contrast_results[genus] = {'diff': est, 'se': se, 't': t, 'p': p}

# Summaries for reporting
coef_human = float(model1.params['human'])
se_human = float(model1.bse['human'])
p_human = float(model1.pvalues['human'])
ci_human = model1.conf_int().loc['human'].tolist()

# Effect size relative to outcome SD
outcome_sd = float(_df['feature3'].std())
std_effect = coef_human / outcome_sd if outcome_sd > 0 else np.nan

results = {
    'n': int(len(_df)),
    'mean_feature3_by_genus': _df.groupby('feature8')['feature3'].mean().to_dict(),
    'adjusted_mean_feature3_by_genus': adj_means,
    'human_vs_nonhuman_coef': coef_human,
    'human_vs_nonhuman_se': se_human,
    'human_vs_nonhuman_p': p_human,
    'human_vs_nonhuman_ci': [float(ci_human[0]), float(ci_human[1])],
    'human_vs_nonhuman_std_effect': std_effect,
    'homo_vs_each_genus_contrasts': contrast_results,
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
