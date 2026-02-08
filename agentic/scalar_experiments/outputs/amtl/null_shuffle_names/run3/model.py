import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
raw = pd.read_csv('amtl.csv')

# Map shuffled columns to semantics
# Based on distributions:
# sockets -> tooth_class
# tooth_class -> genus
# genus -> num_missing
# age -> sockets (total observable)
# pop -> age_years
# stdev_age -> prob_male
# num_amtl -> age_uncertainty (not used)

df = pd.DataFrame({
    'tooth_class': raw['sockets'],
    'genus': raw['tooth_class'],
    'num_missing': raw['genus'],
    'num_sockets': raw['age'],
    'age_years': raw['pop'],
    'prob_male': raw['stdev_age'],
})

# filter invalid rows
valid = df[(df['num_missing'] <= df['num_sockets']) & (df['num_sockets'] > 0)].copy()

# indicator for humans
valid['is_human'] = (valid['genus'] == 'Homo sapiens').astype(int)

# Build GLM binomial with counts
# Use formula with categorical tooth_class and genus indicator
# Endog as proportion with weights is more stable for statsmodels
valid['missing_rate'] = valid['num_missing'] / valid['num_sockets']

formula = 'missing_rate ~ is_human + age_years + prob_male + C(tooth_class)'
model = smf.glm(formula=formula, data=valid, family=sm.families.Binomial(), freq_weights=valid['num_sockets'])
res = model.fit()

print(res.summary())

coef = res.params['is_human']
se = res.bse['is_human']
z = coef / se
p = res.pvalues['is_human']

# Also compute predicted difference at mean covariates
mean_age = valid['age_years'].mean()
mean_sex = valid['prob_male'].mean()
# choose reference tooth_class: alphabetical? statsmodels uses first in sorted for C()
# Let's compute predicted probabilities for each tooth_class averaged
classes = sorted(valid['tooth_class'].unique())

preds = []
for tc in classes:
    for is_human in [0,1]:
        row = {
            'is_human': is_human,
            'age_years': mean_age,
            'prob_male': mean_sex,
            'tooth_class': tc,
        }
        pred = res.predict(pd.DataFrame([row]))[0]
        preds.append((tc, is_human, pred))

# average over tooth_class
pred_human = np.mean([p for tc, ih, p in preds if ih==1])
pred_non = np.mean([p for tc, ih, p in preds if ih==0])

print('coef_is_human', coef)
print('p_is_human', p)
print('pred_human', pred_human, 'pred_non', pred_non, 'diff', pred_human-pred_non)
print('n_valid', len(valid), 'n_total', len(df))

