import pandas as pd
import numpy as np

_df = pd.read_csv('amtl.csv')

# map names as before
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

# correlations for age_uncert with others
print('corr age_uncert with observable_sockets', _df['age_uncert'].corr(_df['observable_sockets']))
print('corr age_uncert with age_est', _df['age_uncert'].corr(_df['age_est']))
print('age_uncert range', _df['age_uncert'].min(), _df['age_uncert'].max())

# is age_uncert integer-ish?
print('age_uncert near integer proportion:', np.mean(np.isclose(_df['age_uncert'], np.round(_df['age_uncert']))))

# check distribution by genus
print(_df.groupby('genus')['age_uncert'].agg(['mean','std','count']))

