import pandas as pd
import numpy as np
from scipy import stats

path = "reading.csv"
df = pd.read_csv(path)

# identify likely reading speed column by checking correlation with computed wpm
# compute wpm from words / reading time (feature5) in ms
# avoid zeros
mask_time = df['feature5'] > 0
wpm = (df.loc[mask_time, 'feature7'] / df.loc[mask_time, 'feature5']) * 60000
# align to original index
wpm_full = pd.Series(index=df.index, dtype=float)
wpm_full.loc[mask_time] = wpm

# correlate wpm with feature20 and maybe inverse with feature5
corr20 = wpm_full.corr(df['feature20'])

print("rows", len(df))
print("wpm stats", wpm_full.describe())
print("feature20 stats", df['feature20'].describe())
print("corr wpm vs feature20", corr20)

# choose speed measure: prefer feature20 if it correlates strongly with computed wpm
# check also correlation with feature5 and feature7
corr_wpm_f5 = wpm_full.corr(df['feature5'])
corr_wpm_f7 = wpm_full.corr(df['feature7'])
print("corr wpm vs feature5", corr_wpm_f5)
print("corr wpm vs feature7", corr_wpm_f7)

# also check feature20 correlations
print("corr feature20 vs feature5", df['feature20'].corr(df['feature5']))
print("corr feature20 vs feature7", df['feature20'].corr(df['feature7']))

# define dyslexia indicator
# feature17 is dyslexia yes/no; feature12 has 0/1/2 severity. We'll use feature17==1

dys = df['feature17'] == 1

# reader view indicator
rv = df['feature3'] == 1

# choose speed variable
# We'll use wpm_full; but if feature20 seems to represent reading speed, we can analyze both.


def compare(speed, label):
    # subset to dyslexia
    data = df.loc[dys].copy()
    data = data.assign(speed=speed)
    data = data[np.isfinite(data['speed'])]
    # group by reader view
    s1 = data.loc[rv & dys, 'speed']
    s0 = data.loc[(~rv) & dys, 'speed']
    # stats
    n1, n0 = len(s1), len(s0)
    mean1, mean0 = s1.mean(), s0.mean()
    sd1, sd0 = s1.std(ddof=1), s0.std(ddof=1)
    # welch t-test
    t_stat, p_val = stats.ttest_ind(s1, s0, equal_var=False, nan_policy='omit')
    # effect size (Cohen's d using pooled SD for unequal n)
    # use Hedges g correction
    sp = np.sqrt(((n1-1)*sd1**2 + (n0-1)*sd0**2) / (n1+n0-2)) if n1+n0-2>0 else np.nan
    d = (mean1-mean0) / sp if sp and sp>0 else np.nan
    # Hedges g
    if np.isfinite(d):
        J = 1 - (3/(4*(n1+n0)-9)) if (n1+n0) > 2 else np.nan
        g = d * J
    else:
        g = np.nan

    # also nonparametric mannwhitney
    try:
        u_stat, p_u = stats.mannwhitneyu(s1, s0, alternative='two-sided')
    except Exception:
        p_u = np.nan

    return {
        'label': label,
        'n1': n1,
        'n0': n0,
        'mean1': mean1,
        'mean0': mean0,
        'sd1': sd1,
        'sd0': sd0,
        't': t_stat,
        'p': p_val,
        'g': g,
        'p_u': p_u,
    }

results = []
results.append(compare(wpm_full, 'computed_wpm'))
results.append(compare(df['feature20'], 'feature20'))

for r in results:
    print("\n", r['label'])
    for k,v in r.items():
        if k=='label':
            continue
        print(k, v)

# Additional analysis: regression with controls (words, language, device) within dyslexia subset
# Using statsmodels
import statsmodels.api as sm

# pick computed wpm as outcome
sub = df.loc[dys].copy()
sub = sub.assign(speed=wpm_full)
sub = sub[np.isfinite(sub['speed'])]

# encode categorical features (device, language)
# limit to main controls: words on page (feature7), readability (feature19), device (feature11), language (feature15), age (feature10)
# add reader view indicator
X = sub[['feature3','feature7','feature19','feature10']].copy()
# one-hot device and language
X = pd.concat([X, pd.get_dummies(sub['feature11'], prefix='device', drop_first=True),
               pd.get_dummies(sub['feature15'], prefix='lang', drop_first=True)], axis=1)
X = sm.add_constant(X)

y = sub['speed']
model = sm.OLS(y, X).fit()
print("\nRegression on computed_wpm (dyslexia subset)")
print(model.summary().tables[1])

# also regression on log speed to reduce skew
sub['log_speed'] = np.log(sub['speed'])
model_log = sm.OLS(sub['log_speed'], X).fit()
print("\nRegression on log(speed)")
print(model_log.summary().tables[1])
