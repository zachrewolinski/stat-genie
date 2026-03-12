import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv('crofoot.csv')

# Derived predictors
_df['rel_size'] = _df['n_focal'] - _df['n_other']
_df['rel_size_ratio'] = _df['n_focal'] / _df['n_other']
_df['loc_adv'] = _df['dist_other'] - _df['dist_focal']  # positive => closer to focal center
_df['loc_adv_binary'] = (_df['dist_focal'] < _df['dist_other']).astype(int)

# Helper for logistic regression

def fit_logit(df, y, xcols):
    X = sm.add_constant(df[xcols])
    model = sm.Logit(df[y], X)
    res = model.fit(disp=False)
    return res

results = {}

# Multivariate model (difference-based)
res_main = fit_logit(_df, 'win', ['rel_size', 'loc_adv'])
results['main'] = res_main

# Alternative model with ratio
res_ratio = fit_logit(_df, 'win', ['rel_size_ratio', 'loc_adv'])
results['ratio'] = res_ratio

# Univariate models
results['rel_size_only'] = fit_logit(_df, 'win', ['rel_size'])
results['loc_adv_only'] = fit_logit(_df, 'win', ['loc_adv'])
results['loc_adv_binary_only'] = fit_logit(_df, 'win', ['loc_adv_binary'])

# Simple descriptive stats
summary = {
    'n': len(_df),
    'win_rate': _df['win'].mean(),
    'rel_size_mean': _df['rel_size'].mean(),
    'loc_adv_mean': _df['loc_adv'].mean(),
}

# Win rates by categories
summary['win_rate_rel_size_pos'] = _df.loc[_df['rel_size'] > 0, 'win'].mean()
summary['win_rate_rel_size_zero'] = _df.loc[_df['rel_size'] == 0, 'win'].mean()
summary['win_rate_rel_size_neg'] = _df.loc[_df['rel_size'] < 0, 'win'].mean()
summary['win_rate_loc_adv_pos'] = _df.loc[_df['loc_adv'] > 0, 'win'].mean()
summary['win_rate_loc_adv_neg'] = _df.loc[_df['loc_adv'] < 0, 'win'].mean()
summary['win_rate_loc_adv_bin1'] = _df.loc[_df['loc_adv_binary'] == 1, 'win'].mean()
summary['win_rate_loc_adv_bin0'] = _df.loc[_df['loc_adv_binary'] == 0, 'win'].mean()

# Print results
print('SUMMARY')
for k, v in summary.items():
    print(f'{k}: {v}')

for name, res in results.items():
    print('\nMODEL:', name)
    print(res.summary())
    # Odds ratios and CI
    params = res.params
    conf = res.conf_int()
    or_df = pd.DataFrame({
        'OR': np.exp(params),
        'CI_low': np.exp(conf[0]),
        'CI_high': np.exp(conf[1]),
        'pvalue': res.pvalues,
    })
    print('\nORs')
    print(or_df)
