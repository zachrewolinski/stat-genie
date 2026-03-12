import pandas as pd
import statsmodels.formula.api as smf

# Load data
raw = pd.read_csv('amtl.csv')

# Rename columns to meaningful names based on metadata distributions
# sockets -> tooth_class (Anterior/Posterior/Premolar)
# prob_male -> specimen_id
# genus -> amtl_measure (number of missing teeth of given class)
# age -> sockets_count (number of observable sockets)
# pop -> age_at_death
# num_amtl -> age_uncertainty (not used here)
# stdev_age -> sex_prob (probability male)
# tooth_class -> genus (Homo sapiens / Pan / Pongo / Papio)
# specimen -> population/region

df = raw.rename(
    columns={
        'sockets': 'tooth_class',
        'prob_male': 'specimen_id',
        'genus': 'amtl',
        'age': 'sockets_count',
        'pop': 'age_at_death',
        'num_amtl': 'age_uncertainty',
        'stdev_age': 'sex_prob',
        'tooth_class': 'genus',
        'specimen': 'population',
    }
)

# Human indicator
_df = df.copy()
_df['human'] = (_df['genus'] == 'Homo sapiens').astype(int)

# Ensure categorical for tooth_class
_df['tooth_class'] = _df['tooth_class'].astype('category')

# Fit OLS with cluster-robust SE by specimen_id (repeated measures per specimen)
model = smf.ols('amtl ~ human + age_at_death + sex_prob + C(tooth_class)', data=_df).fit(
    cov_type='cluster', cov_kwds={'groups': _df['specimen_id']}
)

# Extract coefficient and p-value for human indicator
human_coef = model.params['human']
human_p = model.pvalues['human']

# Adjusted mean difference interpretation (human vs non-human) on amtl scale
# For context, compute raw group means
mean_human = _df.loc[_df['human'] == 1, 'amtl'].mean()
mean_nonhuman = _df.loc[_df['human'] == 0, 'amtl'].mean()

print('n_rows', len(_df))
print('n_specimens', _df['specimen_id'].nunique())
print('mean_amtl_human', mean_human)
print('mean_amtl_nonhuman', mean_nonhuman)
print('human_coef_adj', human_coef)
print('human_p', human_p)
print('model_r2', model.rsquared)
