import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_csv('affairs.csv')

# Clean: ensure children categorical yes/no
# Some datasets might have 'yes'/'no'

# Basic summaries

df['has_affair'] = (df['affairs'] > 0).astype(int)

summary = {}
summary['n'] = len(df)
summary['mean_affairs_overall'] = df['affairs'].mean()
summary['mean_affairs_by_children'] = df.groupby('children')['affairs'].mean().to_dict()
summary['prop_any_affair_by_children'] = df.groupby('children')['has_affair'].mean().to_dict()

# Difference in means
children_groups = df.groupby('children')
mean_yes = children_groups['affairs'].mean().get('yes', np.nan)
mean_no = children_groups['affairs'].mean().get('no', np.nan)
summary['mean_diff_yes_minus_no'] = mean_yes - mean_no

prop_yes = children_groups['has_affair'].mean().get('yes', np.nan)
prop_no = children_groups['has_affair'].mean().get('no', np.nan)
summary['prop_diff_yes_minus_no'] = prop_yes - prop_no

# Regression analyses
# OLS on affairs count (rough), controlling for covariates
# Use C() for categorical variables

formula_ols = 'affairs ~ C(children) + C(gender) + age + yearsmarried + religiousness + education + occupation + rating'
ols_model = smf.ols(formula_ols, data=df).fit()

# Logistic regression for any affair
formula_logit = 'has_affair ~ C(children) + C(gender) + age + yearsmarried + religiousness + education + occupation + rating'
logit_model = smf.logit(formula_logit, data=df).fit(disp=False)

# Extract coefficient for children yes vs no
# statsmodels uses C(children)[T.yes] with 'no' as base if alphabetical? need check

coef_ols = ols_model.params.get('C(children)[T.yes]', np.nan)
se_ols = ols_model.bse.get('C(children)[T.yes]', np.nan)

coef_logit = logit_model.params.get('C(children)[T.yes]', np.nan)
se_logit = logit_model.bse.get('C(children)[T.yes]', np.nan)

summary['ols_coef_children_yes'] = coef_ols
summary['ols_se_children_yes'] = se_ols
summary['ols_p_children_yes'] = ols_model.pvalues.get('C(children)[T.yes]', np.nan)

summary['logit_coef_children_yes'] = coef_logit
summary['logit_se_children_yes'] = se_logit
summary['logit_p_children_yes'] = logit_model.pvalues.get('C(children)[T.yes]', np.nan)

# Convert logit to odds ratio
summary['logit_or_children_yes'] = float(np.exp(coef_logit)) if np.isfinite(coef_logit) else np.nan

print('SUMMARY')
for k, v in summary.items():
    print(f'{k}: {v}')

# Decide conclusion scalar
# Rule of thumb: if having children reduces affairs, expect negative coef and negative diffs
# We'll weight evidence: diff in means and logit/ols coef

score = 0

# mean difference
if np.isfinite(summary['mean_diff_yes_minus_no']):
    if summary['mean_diff_yes_minus_no'] < 0:
        score += 30
    elif summary['mean_diff_yes_minus_no'] > 0:
        score -= 30

# prop difference
if np.isfinite(summary['prop_diff_yes_minus_no']):
    if summary['prop_diff_yes_minus_no'] < 0:
        score += 30
    elif summary['prop_diff_yes_minus_no'] > 0:
        score -= 30

# ols coef
if np.isfinite(summary['ols_coef_children_yes']):
    if summary['ols_coef_children_yes'] < 0:
        score += 20
    elif summary['ols_coef_children_yes'] > 0:
        score -= 20

# logit coef
if np.isfinite(summary['logit_coef_children_yes']):
    if summary['logit_coef_children_yes'] < 0:
        score += 20
    elif summary['logit_coef_children_yes'] > 0:
        score -= 20

# scale score to -100..100
score = max(-100, min(100, int(round(score))))

print('SCORE', score)

with open('conclusion.txt', 'w') as f:
    f.write(str(score))
