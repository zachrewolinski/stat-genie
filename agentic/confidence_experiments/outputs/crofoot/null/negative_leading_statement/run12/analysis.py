import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data

df = pd.read_csv('crofoot.csv')

# Derived predictors
# Relative group size (focal minus other)
df['size_diff'] = df['n_focal'] - df['n_other']
# Contest location advantage: positive means focal is closer to its home range center
# (other distance minus focal distance)
df['loc_adv'] = df['dist_other'] - df['dist_focal']

# Standardize predictors for comparable coefficients
for col in ['size_diff', 'loc_adv']:
    df[col + '_z'] = (df[col] - df[col].mean()) / df[col].std(ddof=0)

# Logistic regression: win ~ size_diff + loc_adv
X = df[['size_diff_z', 'loc_adv_z']]
X = sm.add_constant(X)
model = sm.Logit(df['win'], X)
res = model.fit(disp=False)

# Also check interaction and raw (non-standardized) model for robustness
X_int = df[['size_diff_z', 'loc_adv_z']].copy()
X_int['interaction'] = X_int['size_diff_z'] * X_int['loc_adv_z']
X_int = sm.add_constant(X_int)
res_int = sm.Logit(df['win'], X_int).fit(disp=False)

# Simple group comparisons
win = df['win'] == 1
size_t = stats.ttest_ind(df.loc[win, 'size_diff'], df.loc[~win, 'size_diff'], equal_var=False)
loc_t = stats.ttest_ind(df.loc[win, 'loc_adv'], df.loc[~win, 'loc_adv'], equal_var=False)

# Nonparametric tests
size_u = stats.mannwhitneyu(df.loc[win, 'size_diff'], df.loc[~win, 'size_diff'], alternative='two-sided')
loc_u = stats.mannwhitneyu(df.loc[win, 'loc_adv'], df.loc[~win, 'loc_adv'], alternative='two-sided')

summary = {
    'n': int(len(df)),
    'logit_coef': res.params.to_dict(),
    'logit_pvalues': res.pvalues.to_dict(),
    'logit_or': np.exp(res.params).to_dict(),
    'logit_ci_or': np.exp(res.conf_int()).rename(columns={0:'2.5%',1:'97.5%'}).to_dict(orient='index'),
    'logit_llf': float(res.llf),
    'logit_aic': float(res.aic),
    'logit_pseudo_r2': float(res.prsquared),
    'logit_int_pvalues': res_int.pvalues.to_dict(),
    't_tests': {
        'size_diff_t': float(size_t.statistic),
        'size_diff_p': float(size_t.pvalue),
        'loc_adv_t': float(loc_t.statistic),
        'loc_adv_p': float(loc_t.pvalue),
    },
    'mw_tests': {
        'size_diff_u': float(size_u.statistic),
        'size_diff_p': float(size_u.pvalue),
        'loc_adv_u': float(loc_u.statistic),
        'loc_adv_p': float(loc_u.pvalue),
    },
    'descriptives': {
        'size_diff_mean_win': float(df.loc[win, 'size_diff'].mean()),
        'size_diff_mean_loss': float(df.loc[~win, 'size_diff'].mean()),
        'loc_adv_mean_win': float(df.loc[win, 'loc_adv'].mean()),
        'loc_adv_mean_loss': float(df.loc[~win, 'loc_adv'].mean()),
    }
}

print(json.dumps(summary, indent=2))
