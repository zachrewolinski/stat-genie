import pandas as pd
import statsmodels.api as sm

_df = pd.read_csv('crofoot.csv')
_df['size_diff'] = _df['n_focal'] - _df['n_other']
_df['dist_diff'] = _df['dist_other'] - _df['dist_focal']

X = sm.add_constant(_df[['size_diff','dist_diff']])
res = sm.Logit(_df['win'], X).fit(disp=False)
print(res.summary())
print('\nconf_int:\n', res.conf_int())
