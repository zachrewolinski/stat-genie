import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm


df = pd.read_csv('mortgage.csv')

# helper for two-proportion z-test CI (Wald)

def prop_diff_ci(p1, n1, p0, n0, z=1.96):
    diff = p1 - p0
    se = np.sqrt(p1*(1-p1)/n1 + p0*(1-p0)/n0)
    return diff, (diff - z*se, diff + z*se)

results = {}

for outcome in ['deny','accept']:
    if outcome not in df.columns:
        continue
    y = df[outcome]
    if set(y.dropna().unique()) <= {0,1}:
        sub = df[['female', outcome]].dropna()
        n_f = (sub['female']==1).sum()
        n_m = (sub['female']==0).sum()
        p_f = sub.loc[sub['female']==1, outcome].mean()
        p_m = sub.loc[sub['female']==0, outcome].mean()
        diff, ci = prop_diff_ci(p_f, n_f, p_m, n_m)
        ct = pd.crosstab(sub['female'], sub[outcome])
        chi2, p_chi, dof, exp = stats.chi2_contingency(ct)
        # logistic regression
        X = sm.add_constant(sub[['female']])
        model = sm.Logit(sub[outcome], X)
        res = model.fit(disp=False)
        or_val = np.exp(res.params['female'])
        ci_or = np.exp(res.conf_int().loc['female'])
        results[outcome] = {
            'n_female': int(n_f),
            'n_male': int(n_m),
            'p_female': float(p_f),
            'p_male': float(p_m),
            'diff': float(diff),
            'diff_ci_low': float(ci[0]),
            'diff_ci_high': float(ci[1]),
            'chi2_p': float(p_chi),
            'logit_p': float(res.pvalues['female']),
            'odds_ratio': float(or_val),
            'or_ci_low': float(ci_or[0]),
            'or_ci_high': float(ci_or[1]),
        }

print(results)
