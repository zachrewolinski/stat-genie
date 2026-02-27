import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
path = 'amtl.csv'
df = pd.read_csv(path)

# Create indicator for humans

df['is_human'] = (df['feature8'] == 'Homo sapiens').astype(int)

# OLS with controls
# feature1: tooth class (categorical)
# feature5: age
# feature7: sex estimate

model = smf.ols('feature3 ~ is_human + feature5 + feature7 + C(feature1)', data=df)
res = model.fit(cov_type='cluster', cov_kwds={'groups': df['feature2']})

coef = res.params['is_human']
se = res.bse['is_human']
t = res.tvalues['is_human']
p = res.pvalues['is_human']

# Compute adjusted predictions for human vs non-human at mean covariates
mean_age = df['feature5'].mean()
mean_sex = df['feature7'].mean()
# Use overall distribution of tooth class by weighting predictions
classes = df['feature1'].unique()
class_probs = df['feature1'].value_counts(normalize=True)

preds = {}
for is_human in [0, 1]:
    pred = 0.0
    for cls in classes:
        tmp = pd.DataFrame({
            'is_human': [is_human],
            'feature5': [mean_age],
            'feature7': [mean_sex],
            'feature1': [cls]
        })
        pred += res.predict(tmp)[0] * class_probs[cls]
    preds[is_human] = pred

# Effect size (Cohen's d) using residual SD
resid_sd = res.resid.std(ddof=1)
cohen_d = coef / resid_sd if resid_sd > 0 else np.nan

# Summaries by genus for context
mean_by_genus = df.groupby('feature8')['feature3'].mean()

print('coef_is_human', coef)
print('se', se)
print('t', t)
print('p', p)
print('pred_nonhuman', preds[0])
print('pred_human', preds[1])
print('pred_diff', preds[1] - preds[0])
print('cohen_d', cohen_d)
print('mean_by_genus')
print(mean_by_genus)
