import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
_df = pd.read_csv('amtl.csv')

# Keep relevant columns
cols = ['num_amtl', 'age', 'prob_male', 'tooth_class', 'genus']
_df = _df[cols].copy()

# Drop rows with missing values
_df = _df.dropna()

# Ensure categories
_df['tooth_class'] = _df['tooth_class'].astype('category')
_df['genus'] = _df['genus'].astype('category')

# Set Homo sapiens as reference to compare others against humans
if 'Homo sapiens' in _df['genus'].cat.categories:
    _df['genus'] = _df['genus'].cat.reorder_categories(
        ['Homo sapiens'] + [g for g in _df['genus'].cat.categories if g != 'Homo sapiens'],
        ordered=False
    )

# Linear model (num_amtl appears standardized and can be negative, so OLS is appropriate)
model = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=_df).fit()

# Collect coefficients and p-values for genus comparisons vs Homo sapiens
params = model.params
pvalues = model.pvalues

comparisons = []
for g in _df['genus'].cat.categories:
    if g == 'Homo sapiens':
        continue
    term = f'C(genus)[T.{g}]'
    if term in params:
        comparisons.append({
            'genus': g,
            'coef_vs_homo': params[term],
            'pvalue': pvalues[term]
        })

# Estimate adjusted means for each genus at mean covariates and averaged tooth_class
mean_age = _df['age'].mean()
mean_prob_male = _df['prob_male'].mean()

# Create prediction grid
pred_rows = []
for g in _df['genus'].cat.categories:
    for tc in _df['tooth_class'].cat.categories:
        pred_rows.append({
            'genus': g,
            'tooth_class': tc,
            'age': mean_age,
            'prob_male': mean_prob_male
        })

pred_df = pd.DataFrame(pred_rows)
pred_df['pred'] = model.predict(pred_df)

# Average over tooth_class to get marginal mean per genus
marginal_means = pred_df.groupby('genus')['pred'].mean().to_dict()

# Save results for review
results = {
    'n': int(_df.shape[0]),
    'model_summary': model.summary().as_text(),
    'comparisons': comparisons,
    'marginal_means': marginal_means
}

import json
with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('Wrote analysis_results.json')
