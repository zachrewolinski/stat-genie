import pandas as pd
import statsmodels.api as sm
import numpy as np

# Load data
_df = pd.read_csv('crofoot.csv')

# Define variables
# Outcome: focal won (feature4)
# Relative group size: focal size - other size (feature7 - feature8)
# Contest location: other distance - focal distance (feature6 - feature5)
# Positive location_diff => contest closer to focal home range (focal closer than other)

_df['size_diff'] = _df['feature7'] - _df['feature8']
_df['location_diff'] = _df['feature6'] - _df['feature5']

# Prepare data
_y = _df['feature4']
_X = _df[['size_diff', 'location_diff']]
_X = sm.add_constant(_X)

# Fit logistic regression
_model = sm.Logit(_y, _X)
_result = _model.fit(disp=False)

# Also univariate models
_X_size = sm.add_constant(_df[['size_diff']])
_res_size = sm.Logit(_y, _X_size).fit(disp=False)

_X_loc = sm.add_constant(_df[['location_diff']])
_res_loc = sm.Logit(_y, _X_loc).fit(disp=False)

# Summaries

def _summarize(res):
    params = res.params
    conf = res.conf_int()
    pvals = res.pvalues
    odds = np.exp(params)
    odds_ci = np.exp(conf)
    summary = pd.DataFrame(
        {
            'coef': params,
            'pvalue': pvals,
            'odds_ratio': odds,
            'or_ci_low': odds_ci[0],
            'or_ci_high': odds_ci[1],
        }
    )
    return summary

_summary_full = _summarize(_result)
_summary_size = _summarize(_res_size)
_summary_loc = _summarize(_res_loc)

print('N:', len(_df))
print('\nFull model (size_diff + location_diff):')
print(_summary_full)

print('\nSize-only model:')
print(_summary_size)

print('\nLocation-only model:')
print(_summary_loc)

# Basic descriptive stats
print('\nWin rate:', _y.mean())
print('\nsize_diff stats:', _df['size_diff'].describe())
print('\nlocation_diff stats:', _df['location_diff'].describe())

# Correlation between predictors
print('\nCorrelation size_diff vs location_diff:', _df['size_diff'].corr(_df['location_diff']))
