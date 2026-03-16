import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Create human indicator
_df['human'] = (_df['genus'] == 'Homo sapiens').astype(int)

# Ensure categorical variables
_df['tooth_class'] = _df['tooth_class'].astype('category')
_df['specimen'] = _df['specimen'].astype('category')

# OLS with cluster-robust SE by specimen
ols_model = smf.ols('num_amtl ~ human + age + prob_male + C(tooth_class)', data=_df).fit(
    cov_type='cluster', cov_kwds={'groups': _df['specimen']}
)

print('OLS cluster-robust results')
print(ols_model.summary())

# Mixed effects model with random intercept for specimen
try:
    mixed = smf.mixedlm('num_amtl ~ human + age + prob_male + C(tooth_class)', data=_df, groups=_df['specimen'])
    mixed_fit = mixed.fit(reml=True)
    print('\nMixedLM results')
    print(mixed_fit.summary())
except Exception as e:
    print('MixedLM failed:', e)

# Also compute effect size: difference in adjusted means using model
# We'll compute predicted values for average covariates
mean_age = _df['age'].mean()
mean_prob = _df['prob_male'].mean()

# Use posterior tooth_class? We'll compute average prediction across tooth_class categories
classes = _df['tooth_class'].cat.categories

preds = {}
for human in [0,1]:
    preds_list = []
    for tc in classes:
        row = pd.DataFrame({'human':[human], 'age':[mean_age], 'prob_male':[mean_prob], 'tooth_class':[tc]})
        preds_list.append(ols_model.predict(row)[0])
    preds[human] = np.mean(preds_list)

print('\nAdjusted mean predictions (OLS):')
print('nonhuman', preds[0], 'human', preds[1], 'diff', preds[1]-preds[0])

# compute standardized effect size (Cohen d) using residual SD from OLS
resid_sd = np.sqrt(ols_model.scale)
print('resid_sd', resid_sd, 'cohen_d', (preds[1]-preds[0])/resid_sd)

