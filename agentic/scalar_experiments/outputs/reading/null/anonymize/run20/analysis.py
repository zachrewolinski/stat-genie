import pandas as pd
import numpy as np
from scipy import stats

# Load data
path = 'reading.csv'
df = pd.read_csv(path)

# Basic info
print('rows', len(df), 'cols', df.columns.tolist())

# Identify columns
# feature3: reader view (1=on)
# feature17: dyslexia (1 yes)
# feature12: dyslexia status (0 no,1 dyslexia,2 severe)
# feature5: reading time minus scrolling (ms)
# feature7: word count
# feature20: unknown

# Compute derived reading speed (words per minute) using feature5 reading time
# Avoid divide by zero
speed_wpm = df['feature7'] / (df['feature5'] / 60000.0)
# Replace inf or negative
speed_wpm = speed_wpm.replace([np.inf, -np.inf], np.nan)

# Add to df
DF = df.copy()
DF['speed_wpm'] = speed_wpm

# Compare to feature20 to guess meaning
corr = DF[['speed_wpm','feature20']].corr().iloc[0,1]
print('corr speed_wpm vs feature20', corr)
print('feature20 summary', DF['feature20'].describe())
print('speed_wpm summary', DF['speed_wpm'].describe())

# Focus on dyslexia group using feature17 and feature12
# We'll use feature17==1 for dyslexia

for label, dys_mask in [('feature17', DF['feature17']==1), ('feature12>=1', DF['feature12']>=1)]:
    sub = DF[dys_mask].copy()
    print('\nDyslexia group:', label, 'n=', len(sub))
    # Split by reader view
    on = sub[sub['feature3']==1]
    off = sub[sub['feature3']==0]
    print('n on', len(on), 'n off', len(off))
    # Using speed_wpm
    for metric in ['speed_wpm','feature20']:
        # remove nan
        on_vals = on[metric].dropna()
        off_vals = off[metric].dropna()
        # summary
        print(metric, 'mean on', on_vals.mean(), 'mean off', off_vals.mean())
        # t-test Welch
        tstat, pval = stats.ttest_ind(on_vals, off_vals, equal_var=False)
        # effect size Cohen's d (using pooled sd or hedges g?)
        # We'll use Cohen's d with pooled variance (unequal sample - use standard formula with pooled)
        n1, n2 = len(on_vals), len(off_vals)
        s1, s2 = on_vals.std(ddof=1), off_vals.std(ddof=1)
        sp = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
        d = (on_vals.mean() - off_vals.mean()) / sp if sp>0 else np.nan
        print('t', tstat, 'p', pval, 'cohen d', d)
        # nonparametric test
        try:
            ustat, up = stats.mannwhitneyu(on_vals, off_vals, alternative='two-sided')
            print('mannwhitney p', up)
        except Exception as e:
            print('mannwhitney error', e)

# Also compute linear model controlling for word count, language, device, etc? We can do regression.
# We'll do simple OLS for dyslexia subset: speed_wpm ~ reader_view + word_count + language + device + readability
# Use statsmodels
import statsmodels.formula.api as smf

# Filter dyslexia (feature17==1)
sub = DF[DF['feature17']==1].copy()
# drop nan speed
sub = sub.dropna(subset=['speed_wpm'])

# Some covariates; use feature7 words, feature19 readability, feature10 age, device feature11, language feature15
# Maybe also retake feature16
# Build formula
formula = 'speed_wpm ~ C(feature3) + feature7 + feature19 + feature10 + C(feature11) + C(feature15) + feature16'
model = smf.ols(formula, data=sub).fit()
print('\nOLS dyslexia (feature17==1):')
print(model.summary().tables[1])

# Also use feature20 as DV just in case
sub2 = DF[DF['feature17']==1].copy()
formula2 = 'feature20 ~ C(feature3) + feature7 + feature19 + feature10 + C(feature11) + C(feature15) + feature16'
model2 = smf.ols(formula2, data=sub2).fit()
print('\nOLS dyslexia feature20 DV:')
print(model2.summary().tables[1])

# For robustness, maybe effect across dyslexia severity levels with feature12
sub3 = DF[DF['feature12']>=1].copy()
sub3 = sub3.dropna(subset=['speed_wpm'])
model3 = smf.ols(formula, data=sub3).fit()
print('\nOLS dyslexia (feature12>=1):')
print(model3.summary().tables[1])
