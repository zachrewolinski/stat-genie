import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
path = 'amtl.csv'
df = pd.read_csv(path)

# Create human indicator

df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

# Fit linear model with cluster-robust SE by specimen
model = smf.ols('num_amtl ~ is_human + age + prob_male + C(tooth_class)', data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['specimen']}
)

# Also fit model with genus categories for pairwise comparisons
model_genus = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['specimen']}
)

# Extract key stats
coef = model.params['is_human']
se = model.bse['is_human']
pval = model.pvalues['is_human']
conf_int = model.conf_int().loc['is_human'].tolist()

# Compute adjusted means by genus using model_genus
# Predict at mean age/prob_male and average over tooth_class equally
mean_age = df['age'].mean()
mean_prob_male = df['prob_male'].mean()

levels_tooth = sorted(df['tooth_class'].unique())
levels_genus = sorted(df['genus'].unique())

rows = []
for genus in levels_genus:
    for tc in levels_tooth:
        rows.append({
            'genus': genus,
            'tooth_class': tc,
            'age': mean_age,
            'prob_male': mean_prob_male,
        })

pred_df = pd.DataFrame(rows)
pred_df['pred'] = model_genus.predict(pred_df)

adj_means = pred_df.groupby('genus')['pred'].mean()

# Summaries
print('N rows:', len(df))
print('Unique specimens:', df['specimen'].nunique())
print('\nHuman indicator model (cluster-robust):')
print('coef (is_human):', coef)
print('SE:', se)
print('p-value:', pval)
print('95% CI:', conf_int)

print('\nAdjusted mean predictions by genus (at mean age/prob_male, averaged over tooth_class):')
for genus, val in adj_means.items():
    print(f'{genus}: {val}')

# Save key results for downstream use
results = {
    'coef_is_human': float(coef),
    'se_is_human': float(se),
    'pval_is_human': float(pval),
    'ci_is_human': [float(conf_int[0]), float(conf_int[1])],
    'adj_means_by_genus': {k: float(v) for k, v in adj_means.items()},
}

import json
with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)
