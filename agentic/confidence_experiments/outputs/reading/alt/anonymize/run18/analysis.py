import pandas as pd
import numpy as np
from scipy import stats

# Load data
csv_path = 'reading.csv'
df = pd.read_csv(csv_path)

# Basic info
print('rows', len(df), 'cols', df.shape[1])
print(df.head())

# Compute candidate reading speed (words per minute)
# feature7: number of words; feature5: time spent reading excluding scrolling (ms)
# avoid division by zero
reading_time_min = df['feature5'] / 60000.0
wpm = df['feature7'] / reading_time_min

# Compare with feature20
corr = df['feature20'].corr(wpm)
print('corr feature20 vs wpm', corr)

# summary stats for feature20 and wpm
print('feature20 summary', df['feature20'].describe())
print('wpm summary', wpm.describe())

# maybe feature20 is reading speed, compute correlation with feature7/time
# evaluate if feature20 correlates with wpm near 1

# Focus on dyslexia individuals: feature17 indicates dyslexia (1 yes)
# feature3 indicates reader view

dys = df[df['feature17'] == 1]

# compare reading speed between reader view on/off

for label in ['feature20', 'wpm']:
    if label == 'wpm':
        dys_speed = wpm[dys.index]
    else:
        dys_speed = dys[label]
    on = dys_speed[dys['feature3'] == 1]
    off = dys_speed[dys['feature3'] == 0]
    print('\n', label, 'dyslexia on/off counts', len(on), len(off))
    print('mean on', on.mean(), 'mean off', off.mean())
    # t-test
    t_res = stats.ttest_ind(on, off, equal_var=False, nan_policy='omit')
    print('t-test', t_res)
    # effect size (Cohen d)
    def cohen_d(a, b):
        a = a.dropna()
        b = b.dropna()
        n1, n2 = len(a), len(b)
        v1, v2 = a.var(ddof=1), b.var(ddof=1)
        s = ((n1-1)*v1 + (n2-1)*v2) / (n1 + n2 - 2)
        if s <= 0 or (n1+n2-2) <= 0:
            return np.nan
        return (a.mean() - b.mean()) / np.sqrt(s)
    print('cohen_d', cohen_d(on, off))

# Also run regression controlling for confounders? maybe include word count, language, etc

# Use feature20 as reading speed if correlated; else wpm
if abs(corr) > 0.9:
    speed = df['feature20']
    speed_label = 'feature20'
else:
    speed = wpm
    speed_label = 'wpm'

# Build simple regression: speed ~ reader_view + words + language + age + device + education + dyslexia? but subset to dyslexia
# For question, we look dyslexia only; maybe control for word count and text difficulty

# Use dyslexia subset
sub = dys.copy()
sub = sub.assign(speed=speed.loc[sub.index])

# control variables: word count feature7, readability feature19, language feature15, device feature11, age feature10

# Use statsmodels with categorical dummies
import statsmodels.formula.api as smf

# Prepare formula
formula = 'speed ~ C(feature3) + feature7 + feature19 + feature10 + C(feature11) + C(feature15)'

# Drop missing
sub2 = sub.dropna(subset=['speed', 'feature3', 'feature7', 'feature19', 'feature10', 'feature11', 'feature15'])

model = smf.ols(formula, data=sub2).fit()
print('\nRegression on dyslexia subset, speed label:', speed_label)
print(model.summary())

