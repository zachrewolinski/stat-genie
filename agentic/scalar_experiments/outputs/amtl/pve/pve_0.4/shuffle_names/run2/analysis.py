import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
_df = pd.read_csv('amtl.csv')

# Map columns based on metadata and value inspection
# missing teeth count by tooth class (noisy, can be slightly <0 or > sockets)
_df = _df.rename(columns={
    'genus': 'missing_raw',
    'age': 'sockets_n',
    'pop': 'age_at_death',
    'stdev_age': 'sex_prob_male',
    'tooth_class': 'genus_cat',
    'sockets': 'tooth_class'
})

# Clip missing counts to valid range [0, sockets_n]
_df['missing'] = _df['missing_raw'].clip(lower=0)
_df['missing'] = np.minimum(_df['missing'], _df['sockets_n'])

# Compute proportion for binomial model
_df['missing_prop'] = _df['missing'] / _df['sockets_n']

# Fit binomial GLM with weights
formula = 'missing_prop ~ C(genus_cat) + age_at_death + sex_prob_male + C(tooth_class)'
model = smf.glm(
    formula=formula,
    data=_df,
    family=sm.families.Binomial(),
    freq_weights=_df['sockets_n']
).fit()

print(model.summary())

# Set Homo sapiens as reference by reordering category
_df['genus_cat'] = pd.Categorical(_df['genus_cat'], categories=['Homo sapiens', 'Pan', 'Papio', 'Pongo'])
_df['tooth_class'] = pd.Categorical(_df['tooth_class'], categories=['Anterior', 'Posterior', 'Premolar'])

model_ref = smf.glm(
    formula=formula,
    data=_df,
    family=sm.families.Binomial(),
    freq_weights=_df['sockets_n']
).fit()

print('\nModel with Homo sapiens reference:')
print(model_ref.summary())

# Extract coefficients for non-human genera vs Homo
coefs = model_ref.params
pvals = model_ref.pvalues

comparisons = {}
for genus in ['Pan', 'Papio', 'Pongo']:
    key = f'C(genus_cat)[T.{genus}]'
    comparisons[genus] = {
        'coef': coefs.get(key, np.nan),
        'pval': pvals.get(key, np.nan),
        'odds_ratio': np.exp(coefs.get(key, np.nan))
    }

print('\nComparisons vs Homo sapiens (negative coef => lower odds than Homo):')
for genus, stats in comparisons.items():
    print(genus, stats)

# Compute marginal predicted probabilities by genus at mean covariates and averaged tooth class distribution
mean_age = _df['age_at_death'].mean()
mean_sex = _df['sex_prob_male'].mean()

# Use empirical distribution of tooth_class
preds = []
for genus in ['Homo sapiens', 'Pan', 'Papio', 'Pongo']:
    for tooth_class, weight in _df['tooth_class'].value_counts(normalize=True).items():
        row = {
            'genus_cat': genus,
            'age_at_death': mean_age,
            'sex_prob_male': mean_sex,
            'tooth_class': tooth_class
        }
        pred = model_ref.predict(pd.DataFrame([row]))[0]
        preds.append({'genus': genus, 'tooth_class': tooth_class, 'pred': pred, 'weight': weight})

pred_df = pd.DataFrame(preds)
weighted_pred = pred_df.groupby('genus').apply(lambda g: (g['pred'] * g['weight']).sum())
print('\nWeighted predicted missing proportion (at mean age/sex, averaged tooth class):')
print(weighted_pred)

# Save key results for later use
summary = {
    'comparisons': comparisons,
    'weighted_pred': weighted_pred.to_dict(),
    'mean_age': mean_age,
    'mean_sex': mean_sex
}

import json
with open('analysis_results.json', 'w') as f:
    json.dump(summary, f, indent=2)
