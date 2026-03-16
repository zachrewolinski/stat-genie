import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

# Load data
DF = pd.read_csv('amtl.csv')

# Create binary indicator for humans
DF['is_human'] = (DF['genus'] == 'Homo sapiens').astype(int)

# OLS model: human vs non-human with controls
model = smf.ols('num_amtl ~ is_human + age + prob_male + C(tooth_class)', data=DF).fit(cov_type='HC3')

coef = model.params['is_human']
se = model.bse['is_human']
pval = model.pvalues['is_human']
ci_low, ci_high = model.conf_int().loc['is_human']

# Model with full genus categories for reference (non-human baseline)
model_genus = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=DF).fit(cov_type='HC3')

# Predicted adjusted means by genus (using average covariates)
mean_age = DF['age'].mean()
mean_prob_male = DF['prob_male'].mean()
# Use most common tooth_class as baseline to compare? We'll average across tooth_class by weighting.
# Build a design matrix of all genus x tooth_class combos with mean age/sex.
levels_genus = DF['genus'].unique()
levels_tooth = DF['tooth_class'].unique()

rows = []
for g in levels_genus:
    for t in levels_tooth:
        rows.append({'genus': g, 'tooth_class': t, 'age': mean_age, 'prob_male': mean_prob_male})

pred_df = pd.DataFrame(rows)
# Predict and then average across tooth_class equally
pred = model_genus.predict(pred_df)
pred_df['pred'] = pred
adj_means = pred_df.groupby('genus')['pred'].mean().sort_values(ascending=False)

# Save results to a small json-like text for manual review
print('is_human coef', coef)
print('is_human se', se)
print('is_human pval', pval)
print('is_human ci', (ci_low, ci_high))
print('adj_means', adj_means)
