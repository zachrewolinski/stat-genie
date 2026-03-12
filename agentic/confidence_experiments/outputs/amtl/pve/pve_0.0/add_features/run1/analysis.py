import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Keep relevant columns
cols = ['num_amtl', 'genus', 'age', 'prob_male', 'tooth_class']
df = _df[cols].copy()

# Drop missing
df = df.dropna()

# Ensure categories
for col in ['genus', 'tooth_class']:
    df[col] = df[col].astype('category')

# Relevel genus to Homo sapiens as reference
if 'Homo sapiens' in df['genus'].cat.categories:
    df['genus'] = df['genus'].cat.reorder_categories(
        ['Homo sapiens'] + [c for c in df['genus'].cat.categories if c != 'Homo sapiens'],
        ordered=False
    )

# Fit OLS with robust SE
model = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=df).fit(cov_type='HC3')

# Extract coefficients for genus comparisons vs Homo sapiens
coef = model.params
se = model.bse
pvals = model.pvalues

# Build comparisons
comparisons = {}
for genus in df['genus'].cat.categories:
    if genus == 'Homo sapiens':
        continue
    term = f'C(genus)[T.{genus}]'
    if term in coef:
        comparisons[genus] = {
            'coef_other_minus_homo': float(coef[term]),
            'se': float(se[term]),
            'pvalue': float(pvals[term]),
        }

# Compute adjusted mean for each genus at mean covariates and averaged tooth_class
mean_age = df['age'].mean()
mean_prob_male = df['prob_male'].mean()

genera = list(df['genus'].cat.categories)
classes = list(df['tooth_class'].cat.categories)

rows = []
for g in genera:
    for tc in classes:
        rows.append({'genus': g, 'age': mean_age, 'prob_male': mean_prob_male, 'tooth_class': tc})

pred_df = pd.DataFrame(rows)
pred_df['pred'] = model.predict(pred_df)

adj_means = pred_df.groupby('genus')['pred'].mean().to_dict()

# Compare Homo sapiens to average of non-human genera
nonhuman = [g for g in genera if g != 'Homo sapiens']
nonhuman_mean = float(np.mean([adj_means[g] for g in nonhuman]))
homo_mean = float(adj_means.get('Homo sapiens', np.nan))

# Fit an alternative model with Homo vs nonhuman indicator for a direct test
_df2 = df.copy()
_df2['is_homo'] = (df['genus'] == 'Homo sapiens').astype(int)
model2 = smf.ols('num_amtl ~ is_homo + age + prob_male + C(tooth_class)', data=_df2).fit(cov_type='HC3')
coef2 = float(model2.params['is_homo'])
se2 = float(model2.bse['is_homo'])
pval2 = float(model2.pvalues['is_homo'])

out = {
    'n_rows': int(len(df)),
    'genus_counts': df['genus'].value_counts().to_dict(),
    'comparisons_other_minus_homo': comparisons,
    'adj_means': {k: float(v) for k, v in adj_means.items()},
    'homo_minus_nonhuman_adj_mean': float(homo_mean - nonhuman_mean),
    'homo_vs_nonhuman_coef': coef2,
    'homo_vs_nonhuman_pvalue': pval2,
    'model_r2': float(model.rsquared),
}

print(json.dumps(out, indent=2, sort_keys=True))
