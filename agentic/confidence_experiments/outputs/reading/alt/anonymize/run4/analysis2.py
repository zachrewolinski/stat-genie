import pandas as pd
import numpy as np
from scipy import stats

path = '/home/chenwang/stat-genie/agentic/scalar_experiments/outputs/reading/alt/anonymize/run4/reading.csv'
df = pd.read_csv(path)

df['wpm'] = df['feature20']
# Subset dyslexia (feature17==1)
dys = df[df['feature17'] == 1].copy()

# Compute per-participant mean wpm by reader view condition
pivot = dys.pivot_table(index='feature1', columns='feature3', values='wpm', aggfunc='mean')
# Keep participants with both conditions
paired = pivot.dropna()
print('participants dyslexia total', dys['feature1'].nunique())
print('paired participants', len(paired))

# Paired t-test reader view 1 vs 0
if len(paired) > 1:
    t_res = stats.ttest_rel(paired[1], paired[0])
    # Effect size Cohen's d for paired: mean diff / sd diff
    diff = paired[1] - paired[0]
    d = diff.mean() / diff.std(ddof=1)
    print('paired t-test', t_res)
    print('mean wpm reader view', paired[1].mean())
    print('mean wpm control', paired[0].mean())
    print('mean diff', diff.mean())
    print('cohen_d_paired', d)

# Also compute unpaired test at trial level for reference
rv = dys[dys['feature3']==1]['wpm']
ctrl = dys[dys['feature3']==0]['wpm']
print('trial-level means', rv.mean(), ctrl.mean())

# Welch t-test
welch = stats.ttest_ind(rv, ctrl, equal_var=False)
print('welch t-test', welch)

# Nonparametric (paired) Wilcoxon
if len(paired) > 0:
    try:
        w_res = stats.wilcoxon(paired[1], paired[0])
        print('wilcoxon', w_res)
    except Exception as e:
        print('wilcoxon error', e)

# Effect size for trial-level (Hedges g)
# Use pooled SD for equal weighting; and correction
n1, n2 = len(rv), len(ctrl)
mean1, mean2 = rv.mean(), ctrl.mean()
var1, var2 = rv.var(ddof=1), ctrl.var(ddof=1)
sp = np.sqrt(((n1-1)*var1 + (n2-1)*var2)/(n1+n2-2))
if sp > 0:
    g = (mean1-mean2)/sp
    # Hedges correction
    J = 1 - (3/(4*(n1+n2)-9))
    g *= J
    print('hedges_g_trial', g)

# Provide 95% CI for paired mean difference
if len(paired) > 1:
    diff = paired[1] - paired[0]
    mean_diff = diff.mean()
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    dfree = len(diff)-1
    tcrit = stats.t.ppf(0.975, dfree)
    ci = (mean_diff - tcrit*se, mean_diff + tcrit*se)
    print('paired_diff_ci95', ci)
