import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Basic cleaning: ensure no missing key fields
required = ['num_amtl', 'sockets', 'age', 'prob_male', 'tooth_class', 'genus']
_df = _df.dropna(subset=required).copy()

# Guard against zero sockets
_df = _df[_df['sockets'] > 0].copy()

# Create human indicator
_df['is_human'] = (_df['genus'] == 'Homo sapiens').astype(int)

# Ensure tooth_class as category
_df['tooth_class'] = _df['tooth_class'].astype('category')

# GLM binomial with logit: proportion with weights
_df['amtl_rate'] = _df['num_amtl'] / _df['sockets']

# Model: rate ~ is_human + age + prob_male + tooth_class
formula = 'amtl_rate ~ is_human + age + prob_male + C(tooth_class)'
model = smf.glm(formula=formula, data=_df, family=sm.families.Binomial(), var_weights=_df['sockets'])
res = model.fit()

# Extract coefficient for human
coef = res.params['is_human']
se = res.bse['is_human']
pval = res.pvalues['is_human']

# Compute predicted rates for human vs non-human at mean covariates
mean_age = _df['age'].mean()
mean_male = _df['prob_male'].mean()
# Use each tooth class equally for prediction to avoid imbalance
classes = _df['tooth_class'].cat.categories
pred_rows = []
for tc in classes:
    pred_rows.append({'is_human': 0, 'age': mean_age, 'prob_male': mean_male, 'tooth_class': tc})
    pred_rows.append({'is_human': 1, 'age': mean_age, 'prob_male': mean_male, 'tooth_class': tc})
_pred = pd.DataFrame(pred_rows)
_pred['tooth_class'] = _pred['tooth_class'].astype('category')

pred = res.predict(_pred)
non_human_rate = pred[::2].mean()
human_rate = pred[1::2].mean()
rate_diff = human_rate - non_human_rate

# Map evidence to scalar conclusion
# Start with sign based on coefficient (log-odds). Use p-value and effect size to scale.
# Effect size proxy: absolute rate difference (0-1).
# Scale: strong evidence if p<0.01 and |diff|>=0.05
if np.isnan(pval):
    scalar = 0
else:
    direction = 1 if coef > 0 else -1 if coef < 0 else 0
    # base magnitude from p-value
    if pval < 0.001:
        p_score = 1.0
    elif pval < 0.01:
        p_score = 0.85
    elif pval < 0.05:
        p_score = 0.7
    elif pval < 0.1:
        p_score = 0.55
    else:
        p_score = 0.35
    # effect size score from rate diff (cap at 0.15)
    effect_score = min(abs(rate_diff) / 0.15, 1.0)
    magnitude = 100 * (0.6 * p_score + 0.4 * effect_score)
    scalar = int(round(direction * magnitude))

# Write conclusion
with open('conclusion.txt', 'w') as f:
    f.write(str(scalar))

# Print summary for reporting
print('n_rows', len(_df))
print('coef_is_human', coef)
print('se_is_human', se)
print('pval_is_human', pval)
print('pred_rate_non_human', non_human_rate)
print('pred_rate_human', human_rate)
print('rate_diff', rate_diff)
print('scalar', scalar)
