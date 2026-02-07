import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

_df = pd.read_csv('amtl.csv')
_df = _df.rename(columns={
    'genus': 'amtl_missing',
    'age': 'sockets_observed',
    'pop': 'age_at_death',
    'stdev_age': 'prob_male',
    'sockets': 'tooth_class',
    'tooth_class': 'genus'
})

print(_df.dtypes)
print(_df['tooth_class'].head())
print(type(_df['tooth_class'].iloc[0]))

_df['amtl_missing_clipped'] = _df['amtl_missing'].clip(upper=_df['sockets_observed'])
_df['amtl_rate'] = _df['amtl_missing_clipped'] / _df['sockets_observed']
_df['is_human'] = (_df['genus'] == 'Homo sapiens').astype(int)

formula = 'amtl_rate ~ is_human + age_at_death + prob_male + C(tooth_class)'

try:
    model = smf.glm(
        formula=formula,
        data=_df,
        family=sm.families.Binomial(),
        freq_weights=_df['sockets_observed']
    )
    print('glm init ok')
except Exception as e:
    print('glm init error', e)
