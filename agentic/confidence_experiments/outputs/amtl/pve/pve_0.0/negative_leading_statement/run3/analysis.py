import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Keep relevant columns and drop missing
cols = ['num_amtl', 'age', 'prob_male', 'genus', 'tooth_class']
df = _df[cols].dropna().copy()

# Ensure categories
# Explicit treatment coding with Homo sapiens as reference

# Model 1: binary human vs non-human

df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

model_human = smf.ols(
    'num_amtl ~ is_human + age + prob_male + C(tooth_class, Treatment(reference="Anterior"))',
    data=df
).fit(cov_type='HC3')

# Model 2: genus categorical with Homo sapiens reference
model_genus = smf.ols(
    'num_amtl ~ C(genus, Treatment(reference="Homo sapiens")) + age + prob_male + C(tooth_class, Treatment(reference="Anterior"))',
    data=df
).fit(cov_type='HC3')

# Extract key results
coef_human = model_human.params['is_human']
pval_human = model_human.pvalues['is_human']

# Pairwise contrasts vs Homo sapiens
contrast_info = {}
for genus in ['Pan', 'Pongo', 'Papio']:
    term = f'C(genus, Treatment(reference="Homo sapiens"))[T.{genus}]'
    if term in model_genus.params:
        contrast_info[genus] = {
            'coef': float(model_genus.params[term]),
            'pval': float(model_genus.pvalues[term])
        }

# Compute adjusted mean difference using predictive margins
# Set age and prob_male to their means; tooth_class to its distribution (marginal standardization)
mean_age = df['age'].mean()
mean_prob_male = df['prob_male'].mean()

# Build a prediction frame for each genus and tooth_class combination
rows = []
for genus in ['Homo sapiens', 'Pan', 'Pongo', 'Papio']:
    for tooth_class in df['tooth_class'].unique():
        rows.append({
            'genus': genus,
            'tooth_class': tooth_class,
            'age': mean_age,
            'prob_male': mean_prob_male
        })

pred_df = pd.DataFrame(rows)
# Weight by observed tooth_class proportions
class_weights = df['tooth_class'].value_counts(normalize=True)
pred_df['weight'] = pred_df['tooth_class'].map(class_weights)

pred_df['pred'] = model_genus.predict(pred_df)

adj_means = (
    pred_df.groupby('genus')
    .apply(lambda g: np.average(g['pred'], weights=g['weight']))
)

# Save results for inspection
results = {
    'n': int(df.shape[0]),
    'coef_human': float(coef_human),
    'pval_human': float(pval_human),
    'adj_means': {k: float(v) for k, v in adj_means.items()},
    'contrast_vs_human': contrast_info,
    'model_human_r2': float(model_human.rsquared),
    'model_genus_r2': float(model_genus.rsquared)
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
