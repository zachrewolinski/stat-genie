import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Create indicator for human
_df['is_human'] = (_df['genus'] == 'Homo sapiens').astype(int)

# Model 1: human vs all non-human, controlling for covariates
m1 = smf.ols('num_amtl ~ is_human + age + prob_male + C(tooth_class)', data=_df).fit(cov_type='HC3')

# Model 2: genus-specific contrasts, controlling for covariates
m2 = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=_df).fit(cov_type='HC3')

# Extract key results
coef_is_human = m1.params['is_human']
p_is_human = m1.pvalues['is_human']
ci_is_human = m1.conf_int().loc['is_human'].tolist()

# Genus-specific differences relative to Homo sapiens
# With Homo sapiens as reference, negative coefficients imply lower than human
coef_pan = m2.params.get('C(genus)[T.Pan]', np.nan)
coef_papio = m2.params.get('C(genus)[T.Papio]', np.nan)
coef_pongo = m2.params.get('C(genus)[T.Pongo]', np.nan)

p_pan = m2.pvalues.get('C(genus)[T.Pan]', np.nan)
p_papio = m2.pvalues.get('C(genus)[T.Papio]', np.nan)
p_pongo = m2.pvalues.get('C(genus)[T.Pongo]', np.nan)

ci_pan = m2.conf_int().loc['C(genus)[T.Pan]'].tolist()
ci_papio = m2.conf_int().loc['C(genus)[T.Papio]'].tolist()
ci_pongo = m2.conf_int().loc['C(genus)[T.Pongo]'].tolist()

# Marginal means at average covariates, averaged over tooth classes
mean_age = _df['age'].mean()
mean_prob_male = _df['prob_male'].mean()

# Build a prediction frame for each genus and tooth class, then average
rows = []
for genus in _df['genus'].unique():
    for tc in _df['tooth_class'].unique():
        rows.append({'genus': genus, 'age': mean_age, 'prob_male': mean_prob_male, 'tooth_class': tc})

pred_df = pd.DataFrame(rows)
pred_df['pred'] = m2.predict(pred_df)

marginal_means = pred_df.groupby('genus')['pred'].mean().to_dict()

results = {
    'coef_is_human': float(coef_is_human),
    'p_is_human': float(p_is_human),
    'ci_is_human': [float(ci_is_human[0]), float(ci_is_human[1])],
    'coef_pan': float(coef_pan),
    'p_pan': float(p_pan),
    'ci_pan': [float(ci_pan[0]), float(ci_pan[1])],
    'coef_papio': float(coef_papio),
    'p_papio': float(p_papio),
    'ci_papio': [float(ci_papio[0]), float(ci_papio[1])],
    'coef_pongo': float(coef_pongo),
    'p_pongo': float(p_pongo),
    'ci_pongo': [float(ci_pongo[0]), float(ci_pongo[1])],
    'marginal_means': {k: float(v) for k, v in marginal_means.items()},
    'n': int(_df.shape[0])
}

print(json.dumps(results, indent=2))
