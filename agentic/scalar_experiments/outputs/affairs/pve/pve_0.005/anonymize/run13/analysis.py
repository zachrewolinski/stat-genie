import json
import numpy as np
import pandas as pd
from scipy import stats

# Load data

df = pd.read_csv('affairs.csv')

# Focus on children (feature6) and affair frequency (feature2)
cols = ['feature2', 'feature6']
sub = df[cols].dropna()

# Split groups
g_yes = sub[sub['feature6'] == 'yes']['feature2'].astype(float)
g_no = sub[sub['feature6'] == 'no']['feature2'].astype(float)

# Descriptive stats

def describe(x):
    return {
        'n': int(x.shape[0]),
        'mean': float(np.mean(x)),
        'median': float(np.median(x)),
        'std': float(np.std(x, ddof=1)),
    }

summary = {
    'yes': describe(g_yes),
    'no': describe(g_no),
}

# Welch t-test

t_res = stats.ttest_ind(g_yes, g_no, equal_var=False, nan_policy='omit')

# Mann-Whitney U test (two-sided)

mw_res = stats.mannwhitneyu(g_yes, g_no, alternative='two-sided')

# Effect size: Cohen's d and Hedges' g

def cohens_d(a, b):
    na, nb = len(a), len(b)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    sp = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / sp

na, nb = len(g_yes), len(g_no)

d = cohens_d(g_yes, g_no)
# Hedges g small-sample correction
J = 1.0 - (3.0 / (4.0 * (na + nb) - 9.0))
hedges_g = d * J

# Bootstrap CI for mean difference (yes - no)

rng = np.random.default_rng(0)

def bootstrap_mean_diff(a, b, n_boot=4000):
    a = np.asarray(a)
    b = np.asarray(b)
    na, nb = a.shape[0], b.shape[0]
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        sa = rng.choice(a, size=na, replace=True)
        sb = rng.choice(b, size=nb, replace=True)
        diffs[i] = sa.mean() - sb.mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)

mean_diff = float(g_yes.mean() - g_no.mean())
ci_lo, ci_hi = bootstrap_mean_diff(g_yes.values, g_no.values)

results = {
    'summary': summary,
    'mean_diff_yes_minus_no': mean_diff,
    'mean_diff_bootstrap_ci_95': [ci_lo, ci_hi],
    'welch_t': {
        'statistic': float(t_res.statistic),
        'pvalue': float(t_res.pvalue),
    },
    'mannwhitney_u': {
        'statistic': float(mw_res.statistic),
        'pvalue': float(mw_res.pvalue),
    },
    'effect_size': {
        'cohens_d': float(d),
        'hedges_g': float(hedges_g),
    },
}

print(json.dumps(results, indent=2))
