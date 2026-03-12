import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Rename for clarity
_df = _df.rename(columns={
    'feature1': 'tooth_class',
    'feature2': 'specimen_id',
    'feature3': 'missing_teeth',
    'feature4': 'observable_sockets',
    'feature5': 'age',
    'feature6': 'age_uncertainty',
    'feature7': 'sex_estimate',
    'feature8': 'genus',
    'feature9': 'region'
})

# Keep needed columns and drop rows with missing values
cols = ['missing_teeth', 'observable_sockets', 'age', 'sex_estimate', 'tooth_class', 'genus']
df = _df[cols].copy()

df = df.dropna()

# Convert categories
for c in ['tooth_class', 'genus']:
    df[c] = df[c].astype('category')

# OLS model controlling for age, sex, tooth_class, and observable sockets
formula = 'missing_teeth ~ C(genus) + age + sex_estimate + C(tooth_class) + observable_sockets'
model = smf.ols(formula=formula, data=df)
res = model.fit(cov_type='HC3')

# Determine baseline genus used by patsy (alphabetical)
baseline_genus = sorted(df['genus'].cat.categories)[0]
levels = list(df['genus'].cat.categories)

mean_age = df['age'].mean()
mean_sex = df['sex_estimate'].mean()
mean_obs = df['observable_sockets'].mean()

# Marginalize over tooth_class distribution

tooth_dist = df['tooth_class'].value_counts(normalize=True)


def predict_missing_for_genus(genus):
    rows = []
    for tooth_class, weight in tooth_dist.items():
        rows.append({
            'genus': genus,
            'age': mean_age,
            'sex_estimate': mean_sex,
            'tooth_class': tooth_class,
            'observable_sockets': mean_obs,
            'weight': weight
        })
    pred_df = pd.DataFrame(rows)
    preds = res.predict(pred_df)
    return float(np.sum(preds * pred_df['weight']))

pred_means = {g: predict_missing_for_genus(g) for g in levels}

# Build contrasts for Homo sapiens vs others using linear hypothesis
from patsy import dmatrix

base_tooth = sorted(df['tooth_class'].cat.categories)[0]


def design_row(genus):
    tmp = pd.DataFrame({
        'genus': [genus],
        'age': [mean_age],
        'sex_estimate': [mean_sex],
        'tooth_class': [base_tooth],
        'observable_sockets': [mean_obs],
    })
    return dmatrix(res.model.data.design_info.builder, tmp, return_type='dataframe')

comparisons = []
for other in levels:
    if other == 'Homo sapiens':
        continue
    x_h = design_row('Homo sapiens').values[0]
    x_o = design_row(other).values[0]
    diff = x_h - x_o
    ttest = res.t_test(diff)
    comparisons.append({
        'other_genus': other,
        'coef_diff': float(ttest.effect),
        'se_diff': float(ttest.sd),
        't': float(ttest.tvalue),
        'pvalue': float(ttest.pvalue)
    })

# Overall F-test for genus terms
# Using robust covariance, use wald_test for joint hypothesis
param_names = res.params.index.tolist()

genus_params = [i for i, name in enumerate(param_names) if name.startswith('C(genus)')]
if genus_params:
    # Build restriction matrix
    R = np.zeros((len(genus_params), len(param_names)))
    for row_idx, param_idx in enumerate(genus_params):
        R[row_idx, param_idx] = 1.0
    wald = res.wald_test(R)
    genus_test = {'stat': float(wald.statistic), 'df': int(wald.df_denom), 'pvalue': float(wald.pvalue)}
else:
    genus_test = {'stat': None, 'df': None, 'pvalue': None}

summary = {
    'baseline_genus': baseline_genus,
    'n_rows': int(len(df)),
    'pred_means': pred_means,
    'genus_comparisons': comparisons,
    'genus_wald_test': genus_test,
    'coef_table': res.summary2().tables[1].to_dict(orient='index')
}

with open('analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print('Wrote analysis_summary.json')
