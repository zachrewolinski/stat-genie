import pandas as pd
import numpy as np
from scipy import stats

path = "reading.csv"

df = pd.read_csv(path)
print(df.head())
print(df.columns)

# choose dyslexia indicator
if 'dyslexia_bin' in df.columns:
    dys = df['dyslexia_bin']
else:
    dys = df['dyslexia']

# define dyslexic
if dys.dropna().isin([0,1]).all():
    dyslexic = dys == 1
else:
    # treat >0 as dyslexic
    dyslexic = dys > 0

print("dyslexic count", dyslexic.sum(), "total", len(df))

# speed
speed = df['speed']

# subset
sub = df[dyslexic & df['reader_view'].isin([0,1]) & speed.notna()]
print("subset rows", len(sub))

# group stats
for rv in [0,1]:
    g = sub[sub['reader_view']==rv]['speed']
    print("rv", rv, "n", len(g), "mean", g.mean(), "median", g.median(), "std", g.std())

# Welch t-test
rv0 = sub[sub['reader_view']==0]['speed']
rv1 = sub[sub['reader_view']==1]['speed']

t = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')
print("Welch t-test", t)

# log transform (avoid nonpositive)
rv0_log = np.log(rv0[rv0>0])
rv1_log = np.log(rv1[rv1>0])

print("log sizes", len(rv1_log), len(rv0_log))

print("log mean rv1", rv1_log.mean(), "rv0", rv0_log.mean())
print("log Welch", stats.ttest_ind(rv1_log, rv0_log, equal_var=False, nan_policy='omit'))

# effect size (Cohen's d for unequal var? use pooled SD)

n1, n0 = len(rv1), len(rv0)
mean1, mean0 = rv1.mean(), rv0.mean()
var1, var0 = rv1.var(ddof=1), rv0.var(ddof=1)
# pooled SD
sp = np.sqrt(((n1-1)*var1 + (n0-1)*var0) / (n1+n0-2))
cohen_d = (mean1-mean0)/sp if sp>0 else np.nan
print("cohen_d", cohen_d)

# nonparametric test
print("mannwhitney", stats.mannwhitneyu(rv1, rv0, alternative='two-sided'))

# regression with controls? maybe minimal: speed ~ reader_view + num_words + page_id? We'll quick OLS
import statsmodels.api as sm

reg_df = sub[['speed','reader_view','num_words','page_id']].copy()
reg_df = reg_df.dropna()
# encode page_id categorical
reg_df = pd.get_dummies(reg_df, columns=['page_id'], drop_first=True)
X = reg_df.drop(columns=['speed'])
X = sm.add_constant(X)
y = reg_df['speed']
model = sm.OLS(y, X).fit()
print(model.summary().tables[1])

# log-speed regression
reg_df2 = sub[['speed','reader_view','num_words','page_id']].copy()
reg_df2 = reg_df2[reg_df2['speed']>0].dropna()
reg_df2 = pd.get_dummies(reg_df2, columns=['page_id'], drop_first=True)
X2 = sm.add_constant(reg_df2.drop(columns=['speed']))
y2 = np.log(reg_df2['speed'])
model2 = sm.OLS(y2, X2).fit()
print(model2.summary().tables[1])

