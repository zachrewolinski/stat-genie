import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf


df = pd.read_csv('affairs.csv')

# Map children to binary
children_map = {'yes': 1, 'no': 0}
df['children'] = df['feature6'].map(children_map)

# Group stats
summary = df.groupby('children')['feature2'].agg(['count', 'mean', 'std'])

# Welch's t-test
no_affairs = df.loc[df['children'] == 0, 'feature2']
yes_affairs = df.loc[df['children'] == 1, 'feature2']

t_stat, p_value = stats.ttest_ind(yes_affairs, no_affairs, equal_var=False, nan_policy='omit')

# Cohen's d (using pooled SD)
mean_diff = yes_affairs.mean() - no_affairs.mean()
pooled_sd = np.sqrt(((yes_affairs.std(ddof=1) ** 2) + (no_affairs.std(ddof=1) ** 2)) / 2)
cohens_d = mean_diff / pooled_sd if pooled_sd != 0 else np.nan

# Regression with controls (OLS)
model = smf.ols(
    'feature2 ~ children + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10',
    data=df
).fit(cov_type='HC3')

coef = model.params['children']
pval = model.pvalues['children']

# Decide response based on direction and significance
# Base score around 50 (uncertain). Move toward 0 or 100 based on sign, p-value, and effect size.
score = 50

# Use mean difference and regression coefficient to assess direction
neg_direction = (mean_diff < 0) and (coef < 0)
pos_direction = (mean_diff > 0) and (coef > 0)

if neg_direction:
    # Strength increases with lower p-values and larger absolute effect
    if pval < 0.01:
        score = 80
    elif pval < 0.05:
        score = 70
    elif pval < 0.1:
        score = 60
    else:
        score = 45
elif pos_direction:
    # Evidence against decrease
    if pval < 0.01:
        score = 20
    elif pval < 0.05:
        score = 30
    elif pval < 0.1:
        score = 40
    else:
        score = 55
else:
    # Mixed or unclear direction
    score = 50

# Clamp and convert to int
score = int(round(max(0, min(100, score))))

explanation = (
    f"Compared groups by children status (no children n={int(summary.loc[0,'count'])}, "
    f"children n={int(summary.loc[1,'count'])}). Mean affairs score: no children "
    f"{summary.loc[0,'mean']:.3f} vs children {summary.loc[1,'mean']:.3f}, "
    f"difference (children - no) = {mean_diff:.3f} (Cohen's d={cohens_d:.2f}). "
    f"Welch t-test p={p_value:.4f}. In an OLS regression controlling for gender, age, "
    f"years married, religiousness, education, occupation, and marital rating, the "
    f"children indicator coefficient was {coef:.3f} (robust p={pval:.4f}). "
    "A negative, statistically significant coefficient would support a decrease; otherwise evidence is weak or mixed."
)

with open('conclusion.txt', 'w') as f:
    json.dump({'response': score, 'explanation': explanation}, f)

print('Wrote conclusion.txt')
