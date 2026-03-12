import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data
_df = pd.read_csv('soccer.csv')

# Identify skin tone columns (values in {0,0.25,0.5,0.75,1})
skin_cols = []
for col in _df.columns:
    if pd.api.types.is_numeric_dtype(_df[col]):
        vals = _df[col].dropna().unique()
        if len(vals) <= 6:
            # allow tiny float error
            allowed = np.array([0, 0.25, 0.5, 0.75, 1])
            if np.all(np.isin(np.round(vals, 2), allowed)):
                skin_cols.append(col)

# Deduplicate and keep those with 5 unique values
skin_cols = [c for c in skin_cols if _df[c].nunique() >= 4]

# Compute skin tone as mean of available raters
skin = _df[skin_cols].mean(axis=1, skipna=True)

# Identify games/exposure column: integer count, min >=1, zeros proportion 0, max <=100
exp_candidates = []
for col in _df.columns:
    if pd.api.types.is_numeric_dtype(_df[col]):
        s = _df[col].dropna()
        if len(s)==0:
            continue
        if np.allclose(s % 1, 0):
            if s.min() >= 1 and s.max() <= 100 and (s==0).mean() == 0:
                exp_candidates.append(col)

# Choose the exposure column with highest mean (more games)
exp_col = None
if exp_candidates:
    exp_col = sorted(exp_candidates, key=lambda c: _df[c].mean(), reverse=True)[0]

# Identify red card candidate columns: integer, rare events (mean < 0.05), max <=5
red_candidates = []
for col in _df.columns:
    if pd.api.types.is_numeric_dtype(_df[col]):
        s = _df[col].dropna()
        if len(s)==0:
            continue
        if np.allclose(s % 1, 0):
            if s.mean() < 0.05 and s.max() <= 5:
                red_candidates.append(col)

# Choose red card column with highest correlation with exposure (if available)
red_col = None
if red_candidates:
    if exp_col is not None:
        corrs = []
        for col in red_candidates:
            # Pearson correlation for counts
            corr = np.corrcoef(_df[col], _df[exp_col])[0,1]
            corrs.append((col, corr))
        red_col = sorted(corrs, key=lambda x: x[1], reverse=True)[0][0]
    else:
        red_col = red_candidates[0]

# Build analysis dataset
cols_needed = [c for c in [red_col, exp_col] if c is not None]
_df2 = _df.copy()
_df2['skin_tone'] = skin

# Keep rows with skin_tone and red_col
_df2 = _df2.loc[_df2['skin_tone'].notna()]
if red_col is not None:
    _df2 = _df2.loc[_df2[red_col].notna()]
if exp_col is not None:
    _df2 = _df2.loc[_df2[exp_col].notna()]

# Define dark vs light (exclude mid tone)
_df2['skin_group'] = np.where(_df2['skin_tone'] >= 0.75, 'dark', np.where(_df2['skin_tone'] <= 0.25, 'light', 'mid'))
_df_main = _df2[_df2['skin_group'].isin(['dark', 'light'])].copy()

# Compute rates per game if exposure available
if exp_col is not None:
    _df_main['red_rate'] = _df_main[red_col] / _df_main[exp_col]
else:
    _df_main['red_rate'] = _df_main[red_col]

# Group stats
stats_out = {}
for grp in ['dark', 'light']:
    subset = _df_main[_df_main['skin_group']==grp]
    stats_out[grp] = {
        'n': int(len(subset)),
        'mean_red_cards': float(subset[red_col].mean()),
        'mean_rate': float(subset['red_rate'].mean()),
        'any_red_rate': float((subset[red_col] > 0).mean()),
    }

# Hypothesis tests
# t-test on red_rate
x = _df_main[_df_main['skin_group']=='dark']['red_rate']
y = _df_main[_df_main['skin_group']=='light']['red_rate']

# Welch t-test
try:
    tstat, pval = stats.ttest_ind(x, y, equal_var=False)
except Exception:
    tstat, pval = np.nan, np.nan

# Mann-Whitney U
try:
    ustat, upval = stats.mannwhitneyu(x, y, alternative='two-sided')
except Exception:
    ustat, upval = np.nan, np.nan

# Poisson regression with offset if exposure
poisson_res = None
try:
    df_glm = _df2.dropna(subset=['skin_tone', red_col])
    if exp_col is not None:
        df_glm = df_glm[df_glm[exp_col] > 0].copy()
        offset = np.log(df_glm[exp_col])
    else:
        offset = None
    X = sm.add_constant(df_glm['skin_tone'])
    model = sm.GLM(df_glm[red_col], X, family=sm.families.Poisson(), offset=offset)
    poisson_res = model.fit()
except Exception:
    poisson_res = None

# Spearman correlation between skin tone and red cards
rho, spearman_p = stats.spearmanr(_df2['skin_tone'], _df2[red_col])

print('skin_cols', skin_cols)
print('exp_col', exp_col)
print('red_candidates', red_candidates)
print('red_col', red_col)
print('group_stats', stats_out)
print('tstat', tstat, 'p', pval)
print('mannwhitney_p', upval)
print('spearman_rho', rho, 'p', spearman_p)
if poisson_res is not None:
    print('poisson_beta', poisson_res.params['skin_tone'], 'p', poisson_res.pvalues['skin_tone'])

# Also compute sensitivity for each red candidate
print('\nSensitivity by candidate red column')
for cand in red_candidates:
    temp = _df2.copy()
    temp = temp.dropna(subset=['skin_tone', cand])
    temp['skin_group'] = np.where(temp['skin_tone'] >= 0.75, 'dark', np.where(temp['skin_tone'] <= 0.25, 'light', 'mid'))
    temp = temp[temp['skin_group'].isin(['dark','light'])]
    if exp_col is not None:
        temp['rate'] = temp[cand] / temp[exp_col]
    else:
        temp['rate'] = temp[cand]
    x = temp[temp['skin_group']=='dark']['rate']
    y = temp[temp['skin_group']=='light']['rate']
    try:
        tstat2, pval2 = stats.ttest_ind(x, y, equal_var=False)
    except Exception:
        tstat2, pval2 = np.nan, np.nan
    try:
        rho2, sp2 = stats.spearmanr(temp['skin_tone'], temp[cand])
    except Exception:
        rho2, sp2 = np.nan, np.nan
    print(cand, 'mean_dark', x.mean(), 'mean_light', y.mean(), 't_p', pval2, 'spearman_p', sp2)
