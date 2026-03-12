import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Map columns based on observed patterns
_df = _df.rename(columns={
    'sockets': 'tooth_class',  # anterior/posterior/premolar
    'prob_male': 'specimen_id',
    'tooth_class': 'genus_cat',  # Homo sapiens / Pan / Papio / Pongo
    'pop': 'age_at_death',
    'stdev_age': 'prob_male',
    'age': 'total_sockets',
})

# Interpret genus column as log of missing count; convert to missing count
_df['missing_count_raw'] = np.exp(_df['genus'])

# Cap missing count at total sockets to keep proportions in [0,1]
_df['missing_count'] = np.minimum(_df['missing_count_raw'], _df['total_sockets'])

# Proportion missing
_df['prop_missing'] = _df['missing_count'] / _df['total_sockets']

# Binary indicator for humans
_df['is_human'] = (_df['genus_cat'] == 'Homo sapiens').astype(int)

# Fit binomial GLM with human vs nonhuman
formula_bin = 'prop_missing ~ is_human + age_at_death + prob_male + C(tooth_class)'
model_bin = smf.glm(formula=formula_bin, data=_df, family=sm.families.Binomial(), var_weights=_df['total_sockets']).fit()

# Fit binomial GLM with full genus categories
formula_full = 'prop_missing ~ C(genus_cat) + age_at_death + prob_male + C(tooth_class)'
model_full = smf.glm(formula=formula_full, data=_df, family=sm.families.Binomial(), var_weights=_df['total_sockets']).fit()

print('Binary model summary:')
print(model_bin.summary())

print('\nFull genus model summary:')
print(model_full.summary())

# Compute adjusted predicted means for human vs nonhuman using g-computation

tmp_human = _df.copy()
tmp_human['is_human'] = 1
human_preds = model_bin.predict(tmp_human)


tmp_nonhuman = _df.copy()
tmp_nonhuman['is_human'] = 0
nonhuman_preds = model_bin.predict(tmp_nonhuman)

human_mean = human_preds.mean()
nonhuman_mean = nonhuman_preds.mean()

print('\nAdjusted mean prop_missing (human):', human_mean)
print('Adjusted mean prop_missing (nonhuman):', nonhuman_mean)
print('Difference (human - nonhuman):', human_mean - nonhuman_mean)

# Output key stats
print('\nBinary model is_human coef:', model_bin.params['is_human'])
print('Binary model is_human p-value:', model_bin.pvalues['is_human'])

