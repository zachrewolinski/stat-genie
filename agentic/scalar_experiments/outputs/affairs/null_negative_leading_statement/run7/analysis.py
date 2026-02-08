import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
df = pd.read_csv('affairs.csv')

# Basic cleaning
df['children_bin'] = (df['children'].astype(str).str.lower() == 'yes').astype(int)
df['any_affair'] = (df['affairs'] > 0).astype(int)

# Descriptive stats
group_means = df.groupby('children')['affairs'].mean()
group_any = df.groupby('children')['any_affair'].mean()

# OLS on affairs with controls
# Encode gender as binary
df['gender_male'] = (df['gender'].astype(str).str.lower() == 'male').astype(int)

# Control variables
controls = ['children_bin', 'gender_male', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
X = df[controls]
X = sm.add_constant(X)
ols_model = sm.OLS(df['affairs'], X).fit()

# Logistic regression for any affair
logit_model = sm.Logit(df['any_affair'], X).fit(disp=False)

# Collect summary metrics
ols_coef = ols_model.params['children_bin']
ols_p = ols_model.pvalues['children_bin']
logit_coef = logit_model.params['children_bin']
logit_p = logit_model.pvalues['children_bin']

# Average marginal effect for logit
try:
    margeff = logit_model.get_margeff(at='overall').summary_frame().loc['children_bin']
    logit_me = margeff['dy/dx']
    logit_me_p = margeff['Pr(>|z|)']
except Exception:
    logit_me = np.nan
    logit_me_p = np.nan

# Create a simple scoring heuristic for Likert output
# Negative coefficients and meaningful p-values increase the magnitude.
score = 0

# Use mean difference as base effect
mean_diff = group_means.get('yes', np.nan) - group_means.get('no', np.nan)
if np.isfinite(mean_diff):
    # Negative diff => children associated with lower affairs
    score += -mean_diff * 15  # scale

# OLS contribution
if ols_coef < 0:
    score += min(30, abs(ols_coef) * 10)
else:
    score -= min(30, abs(ols_coef) * 10)
if ols_p < 0.05:
    score += 10
elif ols_p < 0.10:
    score += 5

# Logit contribution
if logit_coef < 0:
    score += min(30, abs(logit_coef) * 10)
else:
    score -= min(30, abs(logit_coef) * 10)
if logit_p < 0.05:
    score += 10
elif logit_p < 0.10:
    score += 5

# Marginal effect contribution
if np.isfinite(logit_me):
    if logit_me < 0:
        score += min(20, abs(logit_me) * 100)
    else:
        score -= min(20, abs(logit_me) * 100)
    if np.isfinite(logit_me_p):
        if logit_me_p < 0.05:
            score += 5
        elif logit_me_p < 0.10:
            score += 2

# Clamp score to [-100, 100]
score = max(-100, min(100, score))

# Output key stats for manual interpretation
print('Mean affairs by children:', group_means.to_dict())
print('Mean any_affair by children:', group_any.to_dict())
print('OLS children coef:', ols_coef, 'p:', ols_p)
print('Logit children coef:', logit_coef, 'p:', logit_p)
print('Logit marginal effect:', logit_me, 'p:', logit_me_p)
print('Heuristic score:', score)

# Save score for later use
with open('analysis_score.txt', 'w') as f:
    f.write(str(int(round(score))))
