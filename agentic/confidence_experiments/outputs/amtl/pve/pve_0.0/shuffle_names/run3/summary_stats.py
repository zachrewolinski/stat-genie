import pandas as pd

_df = pd.read_csv('amtl.csv')
_df = _df.rename(columns={
    'sockets':'tooth_class',
    'prob_male':'specimen_id',
    'genus':'amtl_value',
    'age':'observable_sockets',
    'pop':'age_est',
    'num_amtl':'age_uncert',
    'stdev_age':'prob_male',
    'tooth_class':'genus',
    'specimen':'region'
})

_df['human'] = (_df['genus']=='Homo sapiens').astype(int)

human_mean = _df.loc[_df['human']==1,'amtl_value'].mean()
non_mean = _df.loc[_df['human']==0,'amtl_value'].mean()

print('human mean', human_mean)
print('non-human mean', non_mean)
print('difference (human - non)', human_mean - non_mean)

print('\nMeans by genus:')
print(_df.groupby('genus')['amtl_value'].mean())

# check sample sizes
print('\nCounts by genus:')
print(_df['genus'].value_counts())
