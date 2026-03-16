import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

# Load data
_df = pd.read_csv('amtl.csv')

# Create human indicator
_df['is_human'] = (_df['genus'] == 'Homo sapiens').astype(int)

# OLS with robust SE
model = smf.ols('num_amtl ~ is_human + age + prob_male + C(tooth_class)', data=_df).fit(cov_type='HC3')
print(model.summary())

# Extract coefficient
coef = model.params['is_human']
se = model.bse['is_human']
pval = model.pvalues['is_human']
print('is_human coef', coef, 'se', se, 'p', pval)

# Compare means adjusted? Use model to predict at mean age/prob_male and each tooth class; compute average difference
mean_age = _df['age'].mean()
mean_prob_male = _df['prob_male'].mean()

tooth_classes = sorted(_df['tooth_class'].unique())

preds = []
for tc in tooth_classes:
    for is_human in [0,1]:
        row = {
            'is_human': is_human,
            'age': mean_age,
            'prob_male': mean_prob_male,
            'tooth_class': tc,
        }
        preds.append((tc, is_human, float(model.predict(pd.DataFrame([row])))))

print('Predicted values at mean covariates by tooth_class:')
for tc in tooth_classes:
    p0 = [p for p in preds if p[0]==tc and p[1]==0][0][2]
    p1 = [p for p in preds if p[0]==tc and p[1]==1][0][2]
    print(tc, 'nonhuman', p0, 'human', p1, 'diff', p1-p0)

# Alternative model with genus categories for robustness
model2 = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=_df).fit(cov_type='HC3')
print('\nGenus categorical model coefficients:')
print(model2.params)
print(model2.pvalues)
