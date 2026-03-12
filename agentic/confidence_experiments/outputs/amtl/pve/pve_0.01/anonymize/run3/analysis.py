import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('amtl.csv')

# Ensure categorical types
for col in ['feature1', 'feature8']:
    df[col] = df[col].astype('category')

# OLS with genus baseline Homo sapiens
formula = (
    'feature3 ~ C(feature8, Treatment(reference="Homo sapiens")) '
    '+ C(feature1) + feature5 + feature7 + feature4'
)
model = smf.ols(formula, data=df).fit(cov_type='HC3')

# Extract genus coefficients
coef_table = model.summary2().tables[1]
pval_col = 'P>|t|' if 'P>|t|' in coef_table.columns else 'P>|z|'
ci_low_col = '[0.025]' if '[0.025]' in coef_table.columns else '[0.025'
ci_high_col = '0.975]' if '0.975]' in coef_table.columns else '0.975]'

# Genus contrasts vs Homo sapiens
contrast_rows = [
    'C(feature8, Treatment(reference="Homo sapiens"))[T.Pan]',
    'C(feature8, Treatment(reference="Homo sapiens"))[T.Papio]',
    'C(feature8, Treatment(reference="Homo sapiens"))[T.Pongo]'
]

results = {}
for row in contrast_rows:
    if row in coef_table.index:
        results[row] = {
            'coef': float(coef_table.loc[row, 'Coef.']),
            'se': float(coef_table.loc[row, 'Std.Err.']),
            'pval': float(coef_table.loc[row, pval_col]),
            'ci_low': float(coef_table.loc[row, ci_low_col]),
            'ci_high': float(coef_table.loc[row, ci_high_col]),
        }

# Average non-human contrast (Pan, Papio, Pongo) vs Homo
# Contrast vector: average of three genus indicators
param_names = model.params.index.tolist()
contrast = np.zeros(len(param_names))
for genus in ['Pan', 'Papio', 'Pongo']:
    name = f'C(feature8, Treatment(reference="Homo sapiens"))[T.{genus}]'
    if name in param_names:
        contrast[param_names.index(name)] = 1/3

# t_test on contrast: average(non-human - Homo)
if contrast.sum() != 0:
    ttest = model.t_test(contrast)
    avg_diff = float(ttest.effect)
    avg_p = float(ttest.pvalue)
    avg_ci_low = float(ttest.conf_int()[0, 0])
    avg_ci_high = float(ttest.conf_int()[0, 1])
else:
    avg_diff = avg_p = avg_ci_low = avg_ci_high = None

# G-computation: predicted marginal means by genus
# For each row, set genus to target and predict, then average
marginal_means = {}
for genus in df['feature8'].cat.categories:
    df_tmp = df.copy()
    df_tmp['feature8'] = genus
    preds = model.predict(df_tmp)
    marginal_means[genus] = float(np.mean(preds))

output = {
    'n': int(df.shape[0]),
    'model_r2': float(model.rsquared),
    'genus_contrasts_vs_homo': results,
    'avg_nonhuman_vs_homo': {
        'avg_diff': avg_diff,
        'pval': avg_p,
        'ci_low': avg_ci_low,
        'ci_high': avg_ci_high,
    },
    'marginal_means': marginal_means,
}

print(json.dumps(output, indent=2))
