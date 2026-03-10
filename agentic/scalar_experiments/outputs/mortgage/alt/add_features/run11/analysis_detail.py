import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest


df = pd.read_csv('mortgage.csv')

if 'accept' in df.columns:
    y = df['accept']
else:
    y = 1 - df['deny']

female = df['female']
mask = female.notna() & y.notna()
sub = df.loc[mask].copy()
sub['accept'] = y.loc[mask]
sub['female'] = female.loc[mask]
sub = sub[sub['female'].isin([0,1]) & sub['accept'].isin([0,1])]

# Unadjusted z-test
counts = sub.groupby('female')['accept'].sum()
ns = sub.groupby('female')['accept'].count()
count = np.array([counts.get(1,0), counts.get(0,0)])
obs = np.array([ns.get(1,0), ns.get(0,0)])
stat, pval = proportions_ztest(count, obs)
rate_f = counts.get(1,0)/ns.get(1,0)
rate_m = counts.get(0,0)/ns.get(0,0)
rate_diff = rate_f - rate_m


def fit_logit(data, predictors):
    X = data[predictors].copy()
    X = X.dropna()
    y_ = data.loc[X.index, 'accept']
    X = sm.add_constant(X)
    model = sm.Logit(y_, X)
    res = model.fit(disp=False)
    # statsmodels 0.14 uses private robust method for Logit
    robust = res._get_robustcov_results(cov_type='HC1')
    return res, robust

res_unadj, robust_unadj = fit_logit(sub, ['female'])

control_vars = [
    'black','housing_expense_ratio','self_employed','married','mortgage_credit',
    'consumer_credit','bad_history','PI_ratio','loan_to_value','denied_PMI','age','occupation'
]
controls = [c for c in control_vars if c in sub.columns]
res_adj = robust_adj = None
if controls:
    res_adj, robust_adj = fit_logit(sub, ['female'] + controls)


def extract(res, robust=None):
    coef = res.params['female']
    p = res.pvalues['female']
    se = res.bse['female']
    out = {
        'coef': float(coef),
        'se': float(se),
        'p': float(p),
        'odds_ratio': float(np.exp(coef)),
        'nobs': int(res.nobs)
    }
    if robust is not None:
        out['robust_se'] = float(robust.bse['female'])
        out['robust_p'] = float(robust.pvalues['female'])
    return out

output = {
    'n_total': int(len(sub)),
    'female_count': int(ns.get(1,0)),
    'male_count': int(ns.get(0,0)),
    'accept_rate_female': float(rate_f),
    'accept_rate_male': float(rate_m),
    'rate_diff_f_minus_m': float(rate_diff),
    'ztest_stat': float(stat),
    'ztest_pval': float(pval),
    'logit_unadjusted': extract(res_unadj, robust_unadj),
    'controls_used': controls,
}
if res_adj is not None:
    output['logit_adjusted'] = extract(res_adj, robust_adj)

print(json.dumps(output, indent=2))
