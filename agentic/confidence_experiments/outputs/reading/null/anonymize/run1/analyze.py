import pandas as pd
import numpy as np
from scipy import stats
import json

# Load data
csv_path = 'reading.csv'

df = pd.read_csv(csv_path)

# Basic info
n_rows = len(df)

# Identify columns
cols = df.columns.tolist()

# Compute reading speeds
# feature4: time on page ms; feature5: time minus scrolling ms; feature7: number of words
for c in ['feature4','feature5','feature7']:
    if c not in df.columns:
        raise ValueError(f"Missing {c}")

# Avoid divide by zero
speed_excl_scroll = df['feature7'] / (df['feature5'] / 1000.0) * 60.0
speed_incl_scroll = df['feature7'] / (df['feature4'] / 1000.0) * 60.0

# Add to df

df = df.copy()
df['speed_excl_scroll_wpm'] = speed_excl_scroll

df['speed_incl_scroll_wpm'] = speed_incl_scroll

# Compare with feature20 if exists
if 'feature20' in df.columns:
    # compute correlation with speeds (ignore inf)
    tmp = df.replace([np.inf, -np.inf], np.nan)
    corr_excl = tmp[['feature20', 'speed_excl_scroll_wpm']].dropna().corr().iloc[0,1]
    corr_incl = tmp[['feature20', 'speed_incl_scroll_wpm']].dropna().corr().iloc[0,1]
else:
    corr_excl = corr_incl = np.nan

# Define dyslexia subset
# feature17 indicates dyslexia yes/no
# feature12 indicates 0 no, 1 dyslexia, 2 severe

if 'feature17' in df.columns:
    dyslexia_mask = df['feature17'] == 1
elif 'feature12' in df.columns:
    dyslexia_mask = df['feature12'] > 0
else:
    raise ValueError('No dyslexia indicator')

# Ensure reader view column
if 'feature3' not in df.columns:
    raise ValueError('No reader view indicator (feature3)')

# choose speed metric: if feature20 correlates strongly with computed speed, use feature20
speed_var = 'speed_excl_scroll_wpm'

if 'feature20' in df.columns:
    # pick whichever has higher abs correlation with feature20; but if feature20 seems more plausible (e.g., within wpm range), use computed
    # We'll just report correlations and then use computed wpm for interpretability
    pass

# Subset
sub = df[dyslexia_mask].copy()

# Drop non-finite speeds
sub = sub.replace([np.inf, -np.inf], np.nan)

# We'll use speed_excl_scroll_wpm as primary outcome
sub = sub.dropna(subset=['speed_excl_scroll_wpm'])

# Group by reader view
rv_on = sub[sub['feature3'] == 1]['speed_excl_scroll_wpm']
rv_off = sub[sub['feature3'] == 0]['speed_excl_scroll_wpm']

# Basic stats

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

# Mann-Whitney U (nonparametric)
if len(rv_on) > 0 and len(rv_off) > 0:
    try:
        u_stat, p_u = stats.mannwhitneyu(rv_on, rv_off, alternative='two-sided')
    except Exception:
        u_stat, p_u = np.nan, np.nan
else:
    u_stat, p_u = np.nan, np.nan

# Effect size: Cohen's d

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

# Also compute linear regression to adjust for covariates if available
# We'll include age (feature10), device (feature11), language (feature15), education (feature13), gender (feature14), readability (feature19), words (feature7), retake (feature16)

import statsmodels.api as sm
import statsmodels.formula.api as smf

# Prepare data for regression
reg_cols = ['speed_excl_scroll_wpm', 'feature3', 'feature10', 'feature11', 'feature15', 'feature13', 'feature14', 'feature19', 'feature7', 'feature16']
reg = sub.copy()
# Drop missing in reg cols
reg = reg.dropna(subset=[c for c in reg_cols if c in reg.columns])

reg_result = None
coef = se = p_reg = np.nan

if len(reg) > 10:
    # Build formula with categorical variables
    # Use C() for categorical columns if present
    formula_parts = ['speed_excl_scroll_wpm ~ feature3']
    for c in ['feature10', 'feature19', 'feature7', 'feature16']:
        if c in reg.columns:
            formula_parts.append(f'+ {c}')
    for c in ['feature11', 'feature15', 'feature13', 'feature14']:
        if c in reg.columns:
            formula_parts.append(f'+ C({c})')
    formula = ' '.join(formula_parts)
    try:
        reg_result = smf.ols(formula, data=reg).fit()
        coef = reg_result.params.get('feature3', np.nan)
        se = reg_result.bse.get('feature3', np.nan)
        p_reg = reg_result.pvalues.get('feature3', np.nan)
    except Exception:
        pass

# Output summary

out = {
    'n_rows': n_rows,
    'n_dyslexia': int(sub.shape[0]),
    'corr_feature20_speed_excl': corr_excl,
    'corr_feature20_speed_incl': corr_incl,
    'summary_reader_view_on': summary_on,
    'summary_reader_view_off': summary_off,
    'welch_t_p': p_val,
    'mannwhitney_p': p_u,
    'cohens_d': d,
    'reg_coef_feature3': coef,
    'reg_se_feature3': se,
    'reg_p_feature3': p_reg,
    'reg_n': int(len(reg))
}

print(json.dumps(out, indent=2))
