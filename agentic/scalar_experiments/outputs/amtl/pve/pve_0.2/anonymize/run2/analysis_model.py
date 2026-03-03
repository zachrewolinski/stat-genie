import pandas as pd
import statsmodels.formula.api as smf

amtl = pd.read_csv('amtl.csv')

amtl['is_human'] = (amtl['feature8'] == 'Homo sapiens').astype(int)

# Summary by genus
print('Mean feature3 by genus:')
print(amtl.groupby('feature8')['feature3'].agg(['mean','std','count']))

# OLS model with covariates
model = smf.ols('feature3 ~ is_human + feature5 + feature7 + C(feature1)', data=amtl).fit()
robust = model.get_robustcov_results(cov_type='HC3')

print('\nOLS coefficients (robust SEs):')
print(robust.summary())

# Extract human effect by name
names = robust.model.exog_names
params = dict(zip(names, robust.params))
bse = dict(zip(names, robust.bse))
pvalues = dict(zip(names, robust.pvalues))

coef = params['is_human']
se = bse['is_human']
pval = pvalues['is_human']
ci_low = coef - 1.96 * se
ci_high = coef + 1.96 * se

print('\nHuman effect (is_human):')
print('coef', coef, 'se', se, 'pval', pval, 'CI', (ci_low, ci_high))

# Adjusted predicted mean by human status, averaging over tooth class distribution
mean_feature5 = amtl['feature5'].mean()
mean_feature7 = amtl['feature7'].mean()
class_props = amtl['feature1'].value_counts(normalize=True)

preds = {}
for is_human in [0, 1]:
    pred = 0.0
    for cls, prop in class_props.items():
        df = pd.DataFrame({
            'is_human': [is_human],
            'feature5': [mean_feature5],
            'feature7': [mean_feature7],
            'feature1': [cls]
        })
        pred += model.predict(df).iloc[0] * prop
    preds[is_human] = pred

print('\nAdjusted predicted mean (feature3) by human status:')
print(preds)
print('difference (human - nonhuman):', preds[1] - preds[0])

# Genus model with Homo sapiens as reference
model_genus = smf.ols('feature3 ~ C(feature8, Treatment(reference="Homo sapiens")) + feature5 + feature7 + C(feature1)', data=amtl).fit()
robust_genus = model_genus.get_robustcov_results(cov_type='HC3')

print('\nGenus model coefficients (robust SEs):')
print(robust_genus.summary())

names_g = robust_genus.model.exog_names
params_g = dict(zip(names_g, robust_genus.params))
bse_g = dict(zip(names_g, robust_genus.bse))
pvalues_g = dict(zip(names_g, robust_genus.pvalues))

for level in ['Pan', 'Papio', 'Pongo']:
    key = f'C(feature8, Treatment(reference="Homo sapiens"))[T.{level}]'
    if key in params_g:
        coef_g = params_g[key]
        se_g = bse_g[key]
        pval_g = pvalues_g[key]
        ci_low_g = coef_g - 1.96 * se_g
        ci_high_g = coef_g + 1.96 * se_g
        print(f'{level} vs Homo sapiens: coef {coef_g:.4f}, p {pval_g:.4g}, CI ({ci_low_g:.4f}, {ci_high_g:.4f})')

