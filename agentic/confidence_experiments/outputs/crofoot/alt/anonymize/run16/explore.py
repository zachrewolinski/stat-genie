import pandas as pd
import numpy as np
import statsmodels.api as sm

_df = pd.read_csv('crofoot.csv')
_df = _df.rename(columns={
    'feature4':'win',
    'feature5':'focal_dist',
    'feature6':'other_dist',
    'feature7':'focal_size',
    'feature8':'other_size'
})

_df['size_diff'] = _df['focal_size'] - _df['other_size']
_df['size_ratio'] = _df['focal_size'] / _df['other_size']
_df['loc_diff'] = _df['other_dist'] - _df['focal_dist']
_df['loc_ratio'] = _df['focal_dist'] / (_df['focal_dist'] + _df['other_dist'])
_df['loc_ratio_other'] = _df['other_dist'] / (_df['focal_dist'] + _df['other_dist'])
_df['loc_log_ratio'] = np.log(_df['focal_dist'] / _df['other_dist'])

candidates = [
    ('size_diff','loc_diff'),
    ('size_ratio','loc_diff'),
    ('size_diff','loc_ratio'),
    ('size_ratio','loc_ratio'),
    ('focal_size','other_size'),
    ('focal_dist','other_dist'),
    ('focal_size','other_size','loc_diff'),
    ('size_diff','loc_log_ratio'),
]

for cols in candidates:
    X = sm.add_constant(_df[list(cols)])
    try:
        res = sm.Logit(_df['win'], X).fit(disp=False)
    except Exception as e:
        print(cols, 'fit failed', e)
        continue
    print('\n', cols)
    print(res.params)
    print(res.pvalues)
