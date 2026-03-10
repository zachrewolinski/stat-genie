import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('reading.csv')

# Ensure columns exist
needed = ['reader_view','speed','dyslexia','dyslexia_bin']
for col in needed:
    if col not in df.columns:
        print(f"Missing column: {col}")

# Use dyslexia_bin if available; fallback to dyslexia > 0
if 'dyslexia_bin' in df.columns:
    dyslexic = df[df['dyslexia_bin'] == 1].copy()
else:
    dyslexic = df[df['dyslexia'] > 0].copy()

# Drop missing speed/reader_view

dyslexic = dyslexic.dropna(subset=['speed','reader_view'])

# Basic counts
n_total = len(dyslexic)
counts = dyslexic['reader_view'].value_counts().to_dict()

# Split groups
rv1 = dyslexic[dyslexic['reader_view'] == 1]['speed']
rv0 = dyslexic[dyslexic['reader_view'] == 0]['speed']

# Descriptive stats

def desc(s):
    return {
        'n': int(s.shape[0]),
        'mean': float(np.mean(s)),
        'median': float(np.median(s)),
        'std': float(np.std(s, ddof=1)) if s.shape[0] > 1 else float('nan')
    }

rv1_desc = desc(rv1)
rv0_desc = desc(rv0)

# Compare means: Welch t-test
if rv1.shape[0] > 1 and rv0.shape[0] > 1:
    t_stat, t_p = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')
else:
    t_stat, t_p = float('nan'), float('nan')

# Nonparametric Mann-Whitney U (two-sided)
if rv1.shape[0] > 0 and rv0.shape[0] > 0:
    try:
        u_stat, u_p = stats.mannwhitneyu(rv1, rv0, alternative='two-sided')
    except ValueError:
        u_stat, u_p = float('nan'), float('nan')
else:
    u_stat, u_p = float('nan'), float('nan')

# Effect size: Cohen's d

def cohens_d(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.size < 2 or b.size < 2:
        return float('nan')
    na, nb = a.size, b.size
    sa, sb = np.var(a, ddof=1), np.var(b, ddof=1)
    # pooled SD
    sp = np.sqrt(((na-1)*sa + (nb-1)*sb) / (na+nb-2))
    if sp == 0:
        return float('nan')
    return (np.mean(a) - np.mean(b)) / sp

cohen_d = cohens_d(rv1, rv0)

# Regression: log(speed) ~ reader_view + controls (if columns exist)
# Use log1p to handle zeros
# Controls: page_id, num_words, device, age, gender, language, retake_trial, Flesch_Kincaid, correct_rate
controls = []
for c in ['page_id','num_words','device','age','gender','language','retake_trial','Flesch_Kincaid','correct_rate']:
    if c in dyslexic.columns:
        controls.append(c)

formula = 'np.log1p(speed) ~ reader_view'
for c in controls:
    if dyslexic[c].dtype == 'object' or str(dyslexic[c].dtype).startswith('category'):
        formula += f' + C({c})'
    else:
        formula += f' + {c}'

reg_result = None
if dyslexic['reader_view'].nunique() == 2 and dyslexic['speed'].notna().sum() > 10:
    try:
        reg_result = smf.ols(formula, data=dyslexic).fit()
    except Exception as e:
        reg_result = None

out = {
    'n_total_dyslexic': int(n_total),
    'reader_view_counts': counts,
    'rv1_desc': rv1_desc,
    'rv0_desc': rv0_desc,
    't_stat': float(t_stat),
    't_p': float(t_p),
    'u_stat': float(u_stat),
    'u_p': float(u_p),
    'cohen_d': float(cohen_d),
    'reg_formula': formula,
}

if reg_result is not None:
    out.update({
        'reg_beta_reader_view': float(reg_result.params.get('reader_view', np.nan)),
        'reg_p_reader_view': float(reg_result.pvalues.get('reader_view', np.nan)),
        'reg_n': int(reg_result.nobs),
        'reg_r2': float(reg_result.rsquared),
    })

print(json.dumps(out, indent=2))
