import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

amtl = pd.read_csv('amtl.csv')

# Relevant data
cols = ['num_amtl', 'age', 'prob_male', 'tooth_class', 'genus']

# Drop missing
df = amtl[cols].dropna().copy()

# binary indicator for human
df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

# Summary
print('n', len(df))
print(df['genus'].value_counts())

# OLS model with robust SE
model = smf.ols('num_amtl ~ is_human + age + prob_male + C(tooth_class)', data=df).fit(cov_type='HC3')
print(model.summary())

# Compute effect size (Cohen d) for human vs non-human controlling? We'll approximate with standardized coefficient
# Standardize predictors and outcome for effect size
zdf = df.copy()
for col in ['num_amtl', 'age', 'prob_male']:
    zdf[col] = (zdf[col] - zdf[col].mean()) / zdf[col].std(ddof=0)

zmodel = smf.ols('num_amtl ~ is_human + age + prob_male + C(tooth_class)', data=zdf).fit(cov_type='HC3')
print('standardized coef for is_human', zmodel.params['is_human'], 'p', zmodel.pvalues['is_human'])

# Also check difference in adjusted means using marginal effects of is_human
# We'll predict for typical values (mean age, mean prob_male) and tooth_class distribution
mean_age = df['age'].mean()
mean_prob = df['prob_male'].mean()

# Weighted average over tooth_class distribution
classes = df['tooth_class'].unique()
class_weights = df['tooth_class'].value_counts(normalize=True)

preds = {}
for human in [0,1]:
    pred = 0.0
    for cls, w in class_weights.items():
        row = pd.DataFrame({'is_human':[human], 'age':[mean_age], 'prob_male':[mean_prob], 'tooth_class':[cls]})
        pred += model.predict(row)[0] * w
    preds[human] = pred

print('adjusted mean num_amtl non-human', preds[0])
print('adjusted mean num_amtl human', preds[1])
print('difference', preds[1]-preds[0])

# Also check distribution of num_amtl between 0 and sockets for context
within_bounds = ((amtl['num_amtl'] >= 0) & (amtl['num_amtl'] <= amtl['sockets'])).mean()
print('fraction num_amtl between 0 and sockets', within_bounds)
