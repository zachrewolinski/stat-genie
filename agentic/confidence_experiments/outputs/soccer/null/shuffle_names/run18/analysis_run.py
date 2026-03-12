import pandas as pd
import numpy as np
import json
from scipy import stats

csv_path = 'soccer.csv'

df = pd.read_csv(csv_path)

# Basic info
print('shape', df.shape)
print('columns', df.columns.tolist())

# Identify likely skin tone columns (values between 0 and 1 with 5 unique levels)
likely_skin = []
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        uniques = df[col].dropna().unique()
        if len(uniques) <= 6 and df[col].min() >= 0 and df[col].max() <= 1:
            likely_skin.append(col)
print('likely_skin_cols', likely_skin)

# Identify likely red card columns (counts, integer >=0 with moderate max)
likely_red = []
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        if df[col].dropna().between(0, 50).all():
            # check integer-ish
            if np.allclose(df[col].dropna() % 1, 0):
                # not too few unique values
                if df[col].nunique() > 3:
                    likely_red.append(col)
print('likely_red_cols', likely_red)

# Display summary for candidate columns
for col in likely_skin:
    print('\nSkin candidate', col, df[col].describe())
    print('value counts', df[col].value_counts().sort_index().head(10))

for col in likely_red:
    print('\nRed candidate', col, df[col].describe())
    print('value counts', df[col].value_counts().sort_index().head(10))

# Heuristic: find column with many zeros but some positives for red cards
red_scores = []
for col in likely_red:
    series = df[col].dropna()
    red_scores.append((col, (series==0).mean(), series.mean(), series.max(), series.nunique()))

print('\nred_scores')
for item in sorted(red_scores, key=lambda x: (x[1], -x[2])):
    print(item)

# Choose best guess for skin tone and red cards
# Skin tone: choose column with values in {0,0.25,0.5,0.75,1}
skin_col = None
for col in likely_skin:
    uniq = np.sort(df[col].dropna().unique())
    if len(uniq) <= 5 and np.isin(uniq, [0,0.25,0.5,0.75,1]).all():
        skin_col = col
        break

if skin_col is None and likely_skin:
    skin_col = likely_skin[0]

red_col = None
if red_scores:
    # red cards should have many zeros and low mean; choose highest zero proportion
    red_col = sorted(red_scores, key=lambda x: (-x[1], x[2]))[0][0]

print('\nselected skin_col', skin_col, 'red_col', red_col)

if skin_col and red_col:
    # Make binary skin: dark vs light. Use median or threshold 0.5
    skin = df[skin_col]
    red = df[red_col]
    # Define light as <=0.5 and dark as >0.5
    df2 = df[[skin_col, red_col]].dropna()
    df2['dark'] = (df2[skin_col] > 0.5).astype(int)
    # Compare mean red cards
    mean_dark = df2.loc[df2['dark']==1, red_col].mean()
    mean_light = df2.loc[df2['dark']==0, red_col].mean()
    # t-test
    tstat, pval = stats.ttest_ind(df2.loc[df2['dark']==1, red_col], df2.loc[df2['dark']==0, red_col], equal_var=False)
    # Poisson regression? We'll approximate with Mann-Whitney too
    ustat, upval = stats.mannwhitneyu(df2.loc[df2['dark']==1, red_col], df2.loc[df2['dark']==0, red_col], alternative='two-sided')
    # Effect size (Cohen's d)
    x = df2.loc[df2['dark']==1, red_col]
    y = df2.loc[df2['dark']==0, red_col]
    d = (x.mean() - y.mean()) / np.sqrt(((x.var(ddof=1) + y.var(ddof=1)) / 2))
    print('\nresults')
    print('n_dark', len(x), 'n_light', len(y))
    print('mean_dark', mean_dark, 'mean_light', mean_light, 'diff', mean_dark-mean_light)
    print('tstat', tstat, 'pval', pval)
    print('mannwhitney_p', upval)
    print('cohen_d', d)

    # Also check ordinal correlation (Spearman)
    rho, sp_p = stats.spearmanr(df2[skin_col], df2[red_col])
    print('spearman_rho', rho, 'p', sp_p)

