import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm


df = pd.read_csv('affairs.csv')

# Encode children as binary: yes=1, no=0
child_map = {'yes': 1, 'no': 0}
df['children_bin'] = df['children'].map(child_map)

# Basic stats
mean_by_child = df.groupby('children')['affairs'].mean()
std_overall = df['affairs'].std(ddof=1)

mean_yes = mean_by_child.get('yes', np.nan)
mean_no = mean_by_child.get('no', np.nan)

delta = mean_no - mean_yes  # positive means children lowers affairs

# Welch t-test
x_no = df.loc[df['children'] == 'no', 'affairs']
x_yes = df.loc[df['children'] == 'yes', 'affairs']

if len(x_no) > 1 and len(x_yes) > 1:
    t_stat, t_p = stats.ttest_ind(x_no, x_yes, equal_var=False)
else:
    t_p = np.nan

# OLS with controls
controls = ['children_bin', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
# gender as binary: male=1, female=0
if 'gender' in df.columns:
    df['gender_bin'] = (df['gender'] == 'male').astype(int)
    controls.append('gender_bin')

X = df[controls]
X = sm.add_constant(X)
ols = sm.OLS(df['affairs'], X).fit(cov_type='HC3')

# Poisson GLM with robust SE
poisson = sm.GLM(df['affairs'], X, family=sm.families.Poisson()).fit(cov_type='HC3')

# Extract coefficient and p-values for children
ols_coef = ols.params.get('children_bin', np.nan)
ols_p = ols.pvalues.get('children_bin', np.nan)
pois_coef = poisson.params.get('children_bin', np.nan)
pois_p = poisson.pvalues.get('children_bin', np.nan)

# Build scalar score
score = 0.0

# Direction/effect size from mean difference
if std_overall > 0 and not np.isnan(delta):
    effect_component = 40.0 * min(1.0, abs(delta) / std_overall)
    score += effect_component if delta > 0 else -effect_component

# Significance from t-test
if not np.isnan(t_p):
    if delta > 0:
        if t_p < 0.05:
            score += 20
        elif t_p < 0.1:
            score += 10
    elif delta < 0:
        if t_p < 0.05:
            score -= 20
        elif t_p < 0.1:
            score -= 10

# OLS direction/significance
if not np.isnan(ols_p):
    if ols_coef < 0:
        if ols_p < 0.05:
            score += 10
        elif ols_p < 0.1:
            score += 5
    elif ols_coef > 0:
        if ols_p < 0.05:
            score -= 10
        elif ols_p < 0.1:
            score -= 5

# Poisson direction/significance (stronger weight)
if not np.isnan(pois_p):
    if pois_coef < 0:
        if pois_p < 0.05:
            score += 25
        elif pois_p < 0.1:
            score += 12
    elif pois_coef > 0:
        if pois_p < 0.05:
            score -= 25
        elif pois_p < 0.1:
            score -= 12

# Cap score
score = max(-100, min(100, score))

# Round to integer
score_int = int(round(score))

print('mean_affairs_yes', mean_yes)
print('mean_affairs_no', mean_no)
print('delta_no_minus_yes', delta)
print('t_p', t_p)
print('ols_coef', ols_coef, 'ols_p', ols_p)
print('poisson_coef', pois_coef, 'poisson_p', pois_p)
print('score', score_int)

with open('conclusion.txt', 'w', encoding='utf-8') as f:
    f.write(str(score_int))
