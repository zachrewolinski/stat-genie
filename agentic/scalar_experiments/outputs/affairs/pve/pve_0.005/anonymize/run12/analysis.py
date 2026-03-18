import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

DATA_PATH = Path('affairs.csv')

# Load data
_df = pd.read_csv(DATA_PATH)

# Map children to binary
_df['children_yes'] = _df['feature6'].map({'yes': 1, 'no': 0})

# Outcome
_y = _df['feature2']

# Group stats
_grp = _df.groupby('children_yes')['feature2']
_summary = _grp.agg(['count', 'mean', 'std', 'median'])

# Welch t-test
_y_yes = _df.loc[_df['children_yes'] == 1, 'feature2']
_y_no = _df.loc[_df['children_yes'] == 0, 'feature2']
_t_stat, _t_p = stats.ttest_ind(_y_yes, _y_no, equal_var=False, nan_policy='omit')

# Mann-Whitney U test (two-sided)
try:
    _u_stat, _u_p = stats.mannwhitneyu(_y_yes, _y_no, alternative='two-sided')
except ValueError:
    _u_stat, _u_p = np.nan, np.nan

# Cohen's d (using pooled SD)
_n1, _n0 = len(_y_yes), len(_y_no)
_s1, _s0 = _y_yes.std(ddof=1), _y_no.std(ddof=1)
_pooled_sd = np.sqrt(((_n1 - 1) * _s1**2 + (_n0 - 1) * _s0**2) / (_n1 + _n0 - 2))
_d = (_y_yes.mean() - _y_no.mean()) / _pooled_sd if _pooled_sd > 0 else np.nan

# Difference in means CI (Welch)
_mean_diff = _y_yes.mean() - _y_no.mean()
_se_diff = np.sqrt((_s1**2 / _n1) + (_s0**2 / _n0))
_df_welch = (
    (_s1**2 / _n1 + _s0**2 / _n0) ** 2
    / ((_s1**2 / _n1) ** 2 / (_n1 - 1) + (_s0**2 / _n0) ** 2 / (_n0 - 1))
)
_t_crit = stats.t.ppf(0.975, _df_welch) if np.isfinite(_df_welch) else np.nan
_ci_low = _mean_diff - _t_crit * _se_diff if np.isfinite(_t_crit) else np.nan
_ci_high = _mean_diff + _t_crit * _se_diff if np.isfinite(_t_crit) else np.nan

# Regression with controls
_formula = (
    'feature2 ~ children_yes + C(feature3) + feature4 + feature5 + '
    'feature7 + feature8 + feature9 + feature10'
)
_model = smf.ols(_formula, data=_df).fit(cov_type='HC3')
_coef = _model.params.get('children_yes', np.nan)
_pval = _model.pvalues.get('children_yes', np.nan)
_conf_int = _model.conf_int().loc['children_yes'].tolist()

_results = {
    'summary': _summary.to_dict(),
    't_test': {'t': float(_t_stat), 'p': float(_t_p)},
    'mannwhitney': {'u': float(_u_stat), 'p': float(_u_p)},
    'cohens_d': float(_d),
    'mean_diff': float(_mean_diff),
    'mean_diff_ci95': [float(_ci_low), float(_ci_high)],
    'regression': {
        'coef': float(_coef),
        'p': float(_pval),
        'ci95': [float(_conf_int[0]), float(_conf_int[1])],
        'n': int(_model.nobs),
        'r2': float(_model.rsquared),
    },
}

print(json.dumps(_results, indent=2))
