import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

# Load data
_df = pd.read_csv('amtl.csv')

# Create human indicator
_df['is_human'] = (_df['genus'] == 'Homo sapiens').astype(int)

# Fit OLS with robust SEs
model = smf.ols('num_amtl ~ is_human + age + prob_male + C(tooth_class)', data=_df).fit(cov_type='HC3')

# Extract coefficient info
coef = model.params['is_human']
se = model.bse['is_human']
pval = model.pvalues['is_human']

# Predicted means at average covariates
mean_age = _df['age'].mean()
mean_prob_male = _df['prob_male'].mean()
# Use reference tooth class; compute overall predicted difference is coef in linear model
# We'll compute predicted mean for human and non-human at average covariates for each tooth_class and then average
classes = sorted(_df['tooth_class'].unique())

preds = []
for cls in classes:
    base = {'is_human': 0, 'age': mean_age, 'prob_male': mean_prob_male, 'tooth_class': cls}
    human = {'is_human': 1, 'age': mean_age, 'prob_male': mean_prob_male, 'tooth_class': cls}
    preds.append((
        model.predict(pd.DataFrame([base]))[0],
        model.predict(pd.DataFrame([human]))[0]
    ))

pred_nonhuman = float(np.mean([p[0] for p in preds]))
pred_human = float(np.mean([p[1] for p in preds]))

# Additional model with full genus categories for context
model_genus = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=_df).fit(cov_type='HC3')

print('n', len(_df))
print('coef_is_human', coef)
print('se_is_human', se)
print('pval_is_human', pval)
print('pred_nonhuman', pred_nonhuman)
print('pred_human', pred_human)

# Print genus coefficients (relative to reference) for context
print('genus_params')
for k, v in model_genus.params.items():
    if 'C(genus)' in k:
        print(k, v, 'p', model_genus.pvalues[k])
