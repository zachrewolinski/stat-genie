import pandas as pd
import numpy as np
from scipy import stats

# Load data
path = "reading.csv"
df = pd.read_csv(path)

# Basic sanity
print("rows", df.shape[0], "cols", df.shape[1])

# Define dyslexia group using dyslexia_bin if available, else dyslexia>0
if 'dyslexia_bin' in df.columns:
    dys = df['dyslexia_bin'] == 1
else:
    dys = df['dyslexia'].fillna(0) > 0

# Focus on valid speed and reader_view
use = df[dys & df['reader_view'].isin([0,1]) & df['speed'].notna()]

print("dyslexia rows", use.shape[0])
print("reader_view counts", use['reader_view'].value_counts().to_dict())

# Check per-participant presence
if 'uuid' in use.columns:
    per_user = use.groupby('uuid')['reader_view'].nunique()
    both = (per_user == 2).sum()
    only_one = (per_user == 1).sum()
    print("participants", per_user.shape[0], "both conditions", both, "single condition", only_one)

# Summary stats
summary = use.groupby('reader_view')['speed'].agg(['count','mean','median','std'])
print("speed summary\n", summary)

# Log-transform (positive speeds)
use = use[use['speed']>0].copy()
use['log_speed'] = np.log(use['speed'])
log_summary = use.groupby('reader_view')['log_speed'].agg(['count','mean','median','std'])
print("log speed summary\n", log_summary)

# Welch t-test on log speed
rv0 = use.loc[use['reader_view']==0, 'log_speed']
rv1 = use.loc[use['reader_view']==1, 'log_speed']

welch = stats.ttest_ind(rv1, rv0, equal_var=False)
print("Welch t-test log_speed rv1 vs rv0", welch)

# Mann-Whitney U test on raw speed (nonparam)
try:
    mw = stats.mannwhitneyu(
        use.loc[use['reader_view']==1, 'speed'],
        use.loc[use['reader_view']==0, 'speed'],
        alternative='two-sided'
    )
    print("Mann-Whitney U", mw)
except Exception as e:
    print("Mann-Whitney error", e)

# Effect size (Cohen's d) on log speed
mean1 = rv1.mean(); mean0 = rv0.mean()
var1 = rv1.var(ddof=1); var0 = rv0.var(ddof=1)
n1 = rv1.shape[0]; n0 = rv0.shape[0]
pooled = np.sqrt(((n1-1)*var1 + (n0-1)*var0) / (n1+n0-2))
d = (mean1 - mean0) / pooled
print("Cohen d (log speed)", d)

# Simple linear model with page_id fixed effects, if possible
try:
    import statsmodels.formula.api as smf
    # Use log_speed and control for page_id
    model = smf.ols("log_speed ~ reader_view + C(page_id)", data=use).fit(cov_type='HC3')
    print("OLS robust\n", model.summary().tables[1])
except Exception as e:
    print("OLS error", e)

# Mixed effects with random intercept for uuid if enough data
try:
    import statsmodels.formula.api as smf
    if 'uuid' in use.columns:
        # MixedLM requires numeric groups; uuid ok
        m = smf.mixedlm("log_speed ~ reader_view + C(page_id)", data=use, groups=use["uuid"]).fit(reml=False)
        print("MixedLM\n", m.summary().tables[1])
except Exception as e:
    print("MixedLM error", e)
