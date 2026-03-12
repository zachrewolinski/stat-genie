import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('amtl.csv')

# Keep relevant columns
cols = ['num_amtl', 'age', 'prob_male', 'tooth_class', 'genus']
df = df[cols].dropna()

# Ensure categories
for c in ['tooth_class', 'genus']:
    df[c] = df[c].astype('category')

# Create binary Homo vs non-human

df['is_homo'] = (df['genus'] == 'Homo sapiens').astype(int)

# OLS model with genus categories
model_cat = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=df).fit(cov_type='HC3')

# OLS model with binary Homo
model_bin = smf.ols('num_amtl ~ is_homo + age + prob_male + C(tooth_class)', data=df).fit(cov_type='HC3')

print('Model fit complete.')

# Predicted mean num_amtl for each genus (using observed covariates, setting genus to each, averaging)
# Use model_cat to compute average predicted values

def average_pred_for_genus(genus_name: str):
    temp = df.copy()
    temp['genus'] = genus_name
    pred = model_cat.predict(temp)
    return pred.mean()

mean_preds = {g: average_pred_for_genus(g) for g in df['genus'].cat.categories}

# Difference Homo vs average of non-human (Pan, Pongo, Papio)
non_human = [g for g in df['genus'].cat.categories if g != 'Homo sapiens']
mean_non_human = np.mean([mean_preds[g] for g in non_human])
mean_homo = mean_preds['Homo sapiens']

diff_homo_vs_non = mean_homo - mean_non_human

print('\nMean predictions by genus:', mean_preds)
print('Mean non-human:', mean_non_human)
print('Difference Homo - non-human:', diff_homo_vs_non)

# Bootstrap CI for diff
rng = np.random.default_rng(0)
B = 50
boot_diffs = []

for _ in range(B):
    sample_idx = rng.integers(0, len(df), len(df))
    sample = df.iloc[sample_idx].copy()
    sample['tooth_class'] = sample['tooth_class'].astype('category')
    sample['genus'] = sample['genus'].astype('category')
    # Make sure categories are consistent
    sample['genus'] = sample['genus'].cat.set_categories(df['genus'].cat.categories)
    sample['tooth_class'] = sample['tooth_class'].cat.set_categories(df['tooth_class'].cat.categories)
    try:
        m = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=sample).fit()
    except Exception:
        continue
    def avg_pred(g):
        tmp = sample.copy()
        tmp['genus'] = g
        return m.predict(tmp).mean()
    preds = {g: avg_pred(g) for g in df['genus'].cat.categories}
    mean_non = np.mean([preds[g] for g in non_human])
    diff = preds['Homo sapiens'] - mean_non
    boot_diffs.append(diff)

if boot_diffs:
    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
    print('Bootstrap 95% CI for Homo - non-human diff:', (ci_low, ci_high))
else:
    print('Bootstrap failed')

# Save key outputs to a small json-like dict for later use
import json

result = {
    'n': int(len(df)),
    'coef_is_homo': float(model_bin.params['is_homo']),
    'p_is_homo': float(model_bin.pvalues['is_homo']),
    'mean_homo': float(mean_homo),
    'mean_non_human': float(mean_non_human),
    'diff_homo_non': float(diff_homo_vs_non),
}
if boot_diffs:
    result['diff_ci_low'] = float(ci_low)
    result['diff_ci_high'] = float(ci_high)

print('\nRESULT', json.dumps(result, indent=2))
