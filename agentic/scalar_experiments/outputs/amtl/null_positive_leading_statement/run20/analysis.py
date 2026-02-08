import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import patsy

# Load data
info = json.load(open('info.json'))

df = pd.read_csv('amtl.csv')

# Ensure categories
for col in ['tooth_class','genus','specimen','pop']:
    if col in df.columns:
        df[col] = df[col].astype('category')

# Validate rows
valid = df['sockets'].notna() & df['num_amtl'].notna()
valid &= df['sockets'] > 0
valid &= df['num_amtl'] >= 0
valid &= df['num_amtl'] <= df['sockets']

df = df.loc[valid].copy()

# Set reference category for genus to Homo sapiens
if 'Homo sapiens' in df['genus'].cat.categories:
    df['genus'] = df['genus'].cat.reorder_categories(
        ['Homo sapiens'] + [g for g in df['genus'].cat.categories if g != 'Homo sapiens'],
        ordered=False
    )

# Build design matrix with stable design_info
formula = 'C(genus) + age + prob_male + C(tooth_class)'
exog = patsy.dmatrix(formula, data=df, return_type='dataframe')

# Endog as successes/failures
endog = np.column_stack([df['num_amtl'], df['sockets'] - df['num_amtl']])

model = sm.GLM(endog, exog, family=sm.families.Binomial())
res = model.fit()

# Marginal predicted probabilities by genus (average over observed covariates)

genera = list(df['genus'].cat.categories)

pred_means = {}
for g in genera:
    tmp = df.copy()
    tmp['genus'] = g
    tmp_exog = patsy.build_design_matrices([exog.design_info], tmp, return_type='dataframe')[0]
    pred = res.predict(tmp_exog)
    pred_means[g] = float(np.average(pred, weights=tmp['sockets']))

# Differences: Homo sapiens vs each non-human genus
homo = pred_means.get('Homo sapiens')

comparisons = {}
if homo is not None:
    for g in genera:
        if g == 'Homo sapiens':
            continue
        comparisons[g] = homo - pred_means[g]

# Overall average of non-human genera
non_human = [g for g in genera if g != 'Homo sapiens']
if homo is not None and non_human:
    non_human_mean = float(np.mean([pred_means[g] for g in non_human]))
    overall_diff = homo - non_human_mean
else:
    non_human_mean = None
    overall_diff = None

summary = {
    'n_rows': int(len(df)),
    'genera': genera,
    'pred_means': pred_means,
    'comparisons': comparisons,
    'non_human_mean': non_human_mean,
    'overall_diff': overall_diff,
    'coef': res.params.to_dict(),
    'pvals': res.pvalues.to_dict(),
}

print(json.dumps(summary, indent=2))
