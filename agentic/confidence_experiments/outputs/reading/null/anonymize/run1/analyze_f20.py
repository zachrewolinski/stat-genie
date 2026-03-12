import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf


df = pd.read_csv('reading.csv')

# define dyslexia mask
if 'feature17' in df.columns:
    dyslexia_mask = df['feature17'] == 1
elif 'feature12' in df.columns:
    dyslexia_mask = df['feature12'] > 0
else:
    raise ValueError('No dyslexia indicator')

sub = df[dyslexia_mask].copy()

# outcome
outcome = 'feature20'

# reader view
if 'feature3' not in df.columns:
    raise ValueError('No reader view indicator')

# drop missing
sub = sub.dropna(subset=[outcome, 'feature3'])

rv_on = sub[sub['feature3'] == 1][outcome]
rv_off = sub[sub['feature3'] == 0][outcome]


def summary(x):
    return {
        'n': int(x.shape[0]),
        'mean': float(np.mean(x)),
        'median': float(np.median(x)),
        'std': float(np.std(x, ddof=1)) if x.shape[0] > 1 else float('nan')
    }

summary_on = summary(rv_on)
summary_off = summary(rv_off)

# Welch t-test
if len(rv_on) > 1 and len(rv_off) > 1:
    t_stat, p_val = stats.ttest_ind(rv_on, rv_off, equal_var=False, nan_policy='omit')
else:
    t_stat, p_val = np.nan, np.nan

# Mann-Whitney U
if len(rv_on) > 0 and len(rv_off) > 0:
    try:
        u_stat, p_u = stats.mannwhitneyu(rv_on, rv_off, alternative='two-sided')
    except Exception:
        u_stat, p_u = np.nan, np.nan
else:
    u_stat, p_u = np.nan, np.nan

# Cohen's d

def cohens_d(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    sa = np.var(a, ddof=1)
    sb = np.var(b, ddof=1)
    s = np.sqrt(((na-1)*sa + (nb-1)*sb) / (na + nb - 2))
    if s == 0:
        return np.nan
    return (np.mean(a) - np.mean(b)) / s


d = cohens_d(rv_on, rv_off)

# Regression with covariates similar to previous
reg_cols = [outcome, 'feature3', 'feature10', 'feature11', 'feature15', 'feature13', 'feature14', 'feature19', 'feature7', 'feature16']
reg = sub.dropna(subset=[c for c in reg_cols if c in sub.columns]).copy()

coef = se = p_reg = np.nan
reg_n = len(reg)
if reg_n > 10:
    formula_parts = [f'{outcome} ~ feature3']
    for c in ['feature10', 'feature19', 'feature7', 'feature16']:
        if c in reg.columns:
            formula_parts.append(f'+ {c}')
    for c in ['feature11', 'feature15', 'feature13', 'feature14']:
        if c in reg.columns:
            formula_parts.append(f'+ C({c})')
    formula = ' '.join(formula_parts)
    try:
        model = smf.ols(formula, data=reg).fit()
        coef = model.params.get('feature3', np.nan)
        se = model.bse.get('feature3', np.nan)
        p_reg = model.pvalues.get('feature3', np.nan)
    except Exception:
        pass

print({
    'n_dyslexia': int(len(sub)),
    'summary_reader_view_on': summary_on,
    'summary_reader_view_off': summary_off,
    'welch_t_p': p_val,
    'mannwhitney_p': p_u,
    'cohens_d': d,
    'reg_coef_feature3': coef,
    'reg_se_feature3': se,
    'reg_p_feature3': p_reg,
    'reg_n': int(reg_n)
})
