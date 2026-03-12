import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Indicator for humans
_df['is_human'] = (_df['genus'] == 'Homo sapiens').astype(int)

# Fit model comparing humans vs all non-humans
model_h = smf.ols('num_amtl ~ is_human + age + prob_male + C(tooth_class)', data=_df).fit(
    cov_type='cluster', cov_kwds={'groups': _df['specimen']}
)

# Fit model with genus categories for pairwise comparisons
model_g = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=_df).fit(
    cov_type='cluster', cov_kwds={'groups': _df['specimen']}
)

# Extract core stats
coef_h = model_h.params['is_human']
pval_h = model_h.pvalues['is_human']

# Outcome std for standardized effect
outcome_std = _df['num_amtl'].std()
std_effect = coef_h / outcome_std if outcome_std > 0 else np.nan

# Pairwise contrasts: Homo sapiens vs each non-human genus
# By default, statsmodels uses the first category alphabetically as baseline.
# We'll compute differences using model predictions by setting genus level.
levels = sorted(_df['genus'].unique())

# Build a helper to get adjusted mean for a given genus at mean covariates
mean_age = _df['age'].mean()
mean_prob_male = _df['prob_male'].mean()
# Use a reference tooth_class distribution by averaging over observed classes
classes = _df['tooth_class'].unique()

pred_means = {}
for genus in levels:
    # average over tooth classes equally
    preds = []
    for tc in classes:
        row = pd.DataFrame({
            'genus': [genus],
            'age': [mean_age],
            'prob_male': [mean_prob_male],
            'tooth_class': [tc]
        })
        preds.append(float(model_g.predict(row)))
    pred_means[genus] = float(np.mean(preds))

# Compute differences vs Homo sapiens
homo_mean = pred_means['Homo sapiens']
comparisons = {}
for genus in levels:
    if genus == 'Homo sapiens':
        continue
    comparisons[genus] = homo_mean - pred_means[genus]

# Use t-tests for each contrast
# build contrast vectors from model_g params
# Params order from model_g
param_names = model_g.params.index.tolist()

# For C(genus), statsmodels uses treatment coding with baseline first level
# and includes terms like C(genus)[T.Homo sapiens], etc.

contrast_results = {}
for genus in levels:
    if genus == 'Homo sapiens':
        continue
    # contrast: (Homo sapiens mean) - (genus mean)
    # With treatment coding, this equals:
    # if baseline is genus: depends. We'll do prediction-based difference using t_test on linear combination.
    # We'll compute the linear combination by comparing two design rows for given genus values.
    base = pd.DataFrame({
        'genus': ['Homo sapiens'],
        'age': [mean_age],
        'prob_male': [mean_prob_male],
        'tooth_class': [classes[0]],
    })
    other = pd.DataFrame({
        'genus': [genus],
        'age': [mean_age],
        'prob_male': [mean_prob_male],
        'tooth_class': [classes[0]],
    })
    # Build design matrices via patsy using the model's design info
    from patsy import dmatrix
    design_info = model_g.model.data.design_info
    X_base = dmatrix(design_info, base, return_type='dataframe')
    X_other = dmatrix(design_info, other, return_type='dataframe')
    contrast = (X_base - X_other).values.squeeze()
    tt = model_g.t_test(contrast)
    contrast_results[genus] = {
        'diff': float(tt.effect),
        'pvalue': float(tt.pvalue)
    }

# Save results for reporting
results = {
    'coef_is_human': float(coef_h),
    'pval_is_human': float(pval_h),
    'std_effect': float(std_effect),
    'pred_means': pred_means,
    'pairwise_diffs': comparisons,
    'pairwise_tests': contrast_results,
    'n': int(len(_df))
}

import json
with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
