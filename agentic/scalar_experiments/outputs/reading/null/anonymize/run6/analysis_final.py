import pandas as pd
import numpy as np
from scipy import stats


def welch_ttest(a, b):
    a = a.dropna()
    b = b.dropna()
    na = len(a)
    nb = len(b)
    mean_a = a.mean()
    mean_b = b.mean()
    var_a = a.var(ddof=1)
    var_b = b.var(ddof=1)
    # Welch t
    t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
    # Welch-Satterthwaite df
    se = np.sqrt(var_a/na + var_b/nb)
    df = (var_a/na + var_b/nb)**2 / ((var_a**2)/((na**2)*(na-1)) + (var_b**2)/((nb**2)*(nb-1)))
    # 95% CI for mean difference (a-b)
    diff = mean_a - mean_b
    t_crit = stats.t.ppf(0.975, df)
    ci_low = diff - t_crit * se
    ci_high = diff + t_crit * se
    return {
        'n_a': na,
        'n_b': nb,
        'mean_a': mean_a,
        'mean_b': mean_b,
        'diff': diff,
        't_stat': t_stat,
        'p_val': p_val,
        'df': df,
        'ci_low': ci_low,
        'ci_high': ci_high,
    }


def cohen_d(a, b):
    a = a.dropna(); b = b.dropna()
    na = len(a); nb = len(b)
    sa = a.var(ddof=1); sb = b.var(ddof=1)
    s_pooled = np.sqrt(((na-1)*sa + (nb-1)*sb) / (na+nb-2))
    if s_pooled == 0:
        return np.nan
    return (a.mean() - b.mean()) / s_pooled


df = pd.read_csv('reading.csv')

# Derived words per minute based on feature5 (reading time minus scrolling)
# time minutes = ms / 60000

df['wpm_calc'] = df['feature7'] / (df['feature5'] / 60000.0)

# Dyslexia definitions
filters = {
    'feature17==1': df['feature17'] == 1,
    'feature12>0': df['feature12'] > 0,
}

for label, mask in filters.items():
    sub = df[mask]
    print('\nDyslexia subset:', label, 'rows', len(sub))
    for speed_col in ['feature20', 'wpm_calc']:
        g1 = sub[sub['feature3'] == 1][speed_col]
        g0 = sub[sub['feature3'] == 0][speed_col]
        res = welch_ttest(g1, g0)
        d = cohen_d(g1, g0)
        print('  speed:', speed_col)
        print('    n reader view', res['n_a'], 'n no view', res['n_b'])
        print('    mean reader view', res['mean_a'])
        print('    mean no view', res['mean_b'])
        print('    diff (view - no)', res['diff'])
        print('    95% CI', res['ci_low'], res['ci_high'])
        print('    t', res['t_stat'], 'p', res['p_val'])
        print('    cohen d', d)

# Overall correlation checks for context
print('\nCorrelation feature20 vs wpm_calc', df[['feature20','wpm_calc']].corr().iloc[0,1])
